import struct
import numpy as np
import torch
from .WeightFault import WeightFault

class WeightFaultInjector:

    def __init__(self, network):

        self.network = network

        self.layer_name = None
        self.tensor_index = None
        self.bit = None

        self.golden_value = None


    def __inject_fault(self, fault, value=None):
        layer_name, tensor_index, bit = fault.layer_name, fault.tensor_index, fault.bit
        self.layer_name = layer_name
        self.tensor_index = tensor_index
        self.bit = bit
        self.golden_value = float(self.network.state_dict()[self.layer_name + ".weight"][self.tensor_index])

        # If the value is not set, then we are doing a bit-flip
        if value is None:
            faulty_value = self.__float32_bit_flip()
        else:
            faulty_value = self.__float32_stuck_at(value)

        self.faulty_value = faulty_value

        self.network.state_dict()[self.layer_name + ".weight"][self.tensor_index] = faulty_value

    def __float32_bit_flip(self):
        """
        Inject a bit-flip on a data represented as float32
        :return: The value of the bit-flip on the golden value
        """
        float_list = []
        a = struct.pack('!f', self.golden_value)
        b = struct.pack('!I', int(2. ** self.bit))
        for ba, bb in zip(a, b):
            float_list.append(ba ^ bb)

        faulted_value = struct.unpack('!f', bytes(float_list))[0]

        return faulted_value

    def __float32_stuck_at(self,
                           value: int):
        """
        Inject a stuck-at fault on a data represented as float32
        :param value: the value to use as stuck-at value
        :return: The value of the bit-flip on the golden value
        """
        float_list = []
        a = struct.pack('!f', self.golden_value)
        b = struct.pack('!I', int(2. ** self.bit))
        for ba, bb in zip(a, b):
            if value == 1:
                float_list.append(ba | bb)
            else:
                float_list.append(ba & (255 - bb))

        faulted_value = struct.unpack('!f', bytes(float_list))[0]

        return faulted_value

    def restore_golden(self):
        """
        Restore the value of the faulted network weight to its golden value
        """
        if self.layer_name is None:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            quit()

        self.network.state_dict()[self.layer_name + ".weight"][self.tensor_index] = self.golden_value

    def inject_bit_flip(self,
                        fault):
        """
        Inject a bit-flip in the specified layer at the tensor_index position for the specified bit
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault
        """
        self.__inject_fault(fault)

    def inject_stuck_at(self,
                        fault,
                        value: int):
        """
        Inject a stuck-at fault to the specified value in the specified layer at the tensor_index position for the
        specified bit
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault
        :param value: The stuck-at value to set
        """
        self.__inject_fault(fault,
                            value=value)
        
        

def _evaluate_faulty_model(model, device, test_loader, clean_by_batch, injector, fault_site, inj_id,
                    total_samples, baseline_hist, baseline_dist, num_classes):
    ln, ti, bt = fault_site
    fault = WeightFault(injection=inj_id, layer_name=ln, tensor_index=ti, bit=bt)
     
    try:
        injector.inject_bit_flip(fault)
        
        mismatches = 0
        fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
        mism_by_clean = np.zeros(num_classes, dtype=np.int64)
        cnt_by_clean  = np.zeros(num_classes, dtype=np.int64)
        cm_cf = np.zeros((num_classes, num_classes), dtype=np.int64)

        with torch.inference_mode():
            for (batch_i, (xb, _)) in enumerate(test_loader):
                xb = xb.to(device)
                logits_f = model(xb)
                pred_f = torch.argmax(logits_f, dim=1).cpu().numpy()
                np.add.at(fault_hist, pred_f, 1)

                clean_pred = clean_by_batch[batch_i].numpy()
                mism = (pred_f != clean_pred)
                mismatches += int(mism.sum())

                np.add.at(cm_cf, (clean_pred, pred_f), 1)

                for c in range(num_classes):
                    msk = (clean_pred == c)
                    cnt_by_clean[c]  += int(msk.sum())
                    if msk.any():
                        mism_by_clean[c] += int((pred_f[msk] != c).sum())

        frcrit = mismatches / float(total_samples)

        fault_total = max(1, int(fault_hist.sum()))
        fault_dist = fault_hist / fault_total
        maj_cls = int(fault_hist.argmax()) if fault_hist.size > 0 else -1
        maj_share = float(fault_hist.max()) / fault_total if fault_hist.size > 0 else 0.0
        delta_max = float(np.max(np.abs(fault_dist - baseline_dist))) if fault_hist.size > 0 else 0.0
        eps = 1e-12
        kl = float(np.sum(fault_dist * np.log((fault_dist + eps) / (baseline_dist + eps)))) if fault_hist.size > 0 else 0.0

        off_sum = int(cm_cf.sum() - np.trace(cm_cf))
        asym_num = 0
        if num_classes >= 2:
            diff = np.abs(cm_cf - cm_cf.T)
            asym_num = int(diff[np.triu_indices(num_classes, k=1)].sum())
        flip_asym = float(asym_num) / max(1, off_sum)
        agree = float(np.trace(cm_cf)) / max(1, int(cm_cf.sum()))

        bias = {
            "maj_cls": maj_cls,
            "maj_share": maj_share,
            "delta_max": delta_max,
            "kl": kl,
            "flip_asym": flip_asym,
            "agree": agree
        }

    finally:
        injector.restore_golden()
    
    return frcrit, fault, bias, fault_hist, mism_by_clean, cnt_by_clean, cm_cf