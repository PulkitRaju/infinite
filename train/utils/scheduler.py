import numpy as np

class TriageState:
    def __init__(self, domain_ids, beta, init_acc_ema):
        # self.acc_ema: dict[domain_id, float]
        # self.beta = beta
      
        self.beta = float(beta)
        if not (0.0 < float(beta) <= 1.0):
            raise ValueError("beta must be in (0, 1].")
        if not (0.0 <= float(init_acc_ema) <= 1.0):
            raise ValueError("init_acc_ema must be in [0, 1].")
        domain_ids = list(dict.fromkeys(str(d).strip() for d in (domain_ids or [])))
        self.init_acc_ema = float(init_acc_ema)
        self.acc_ema = {domain_id: init_acc_ema for domain_id in domain_ids}
       

    def update_acc(self, domain_id, passed_bool: bool):
        # acc <- (1 - beta)*acc + beta*pass
        domain_id = str(domain_id).strip()
        passed_bool = 1.0 if bool(passed_bool) else 0.0
        # if domain_id not in self.acc_ema:
        #     raise ValueError(f"Domain {domain_id} not found in acc_ema.")
        if domain_id not in self.acc_ema:
            self.acc_ema[domain_id] = self.init_acc_ema
        self.acc_ema[domain_id] = min(1.0, max(0.0, (1 - self.beta) * self.acc_ema[domain_id] + self.beta * passed_bool)) #clamping acc_ema
        
    def state_dict(self,version=1) -> dict:
        return {
        "acc_ema": dict(self.acc_ema),  # shallow copy is fine here
        "beta": float(self.beta),
        "init_acc_ema": float(self.init_acc_ema), 
        "version": version
        }
        

    def load_state_dict(self, state: dict):
        self.acc_ema = dict(state["acc_ema"])
        self.beta = float(state["beta"])
        self.init_acc_ema = float(state["init_acc_ema"])
        self.version = int(state["version"])


class BitterScheduler:
    def __init__(self, domain_ids, beta=0.05, tau=3.0, eps=1e-3, init_acc_ema=0.5, seed=0):
        # self.tau, self.eps; self.rng; self.state = TriageState(...)
        self.tau = tau
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.state = TriageState(domain_ids, beta, init_acc_ema)

    def get_domain_probs(self) -> dict[str, float]:
        # logits_i = tau * (1 - acc_ema_i)
        # probs = softmax(logits)
        # p_i = max(probs_i, eps); renormalize to sum=1
        raise NotImplementedError("Not implemented.")
    def sample_domains(self, k: int) -> list[str]:
        # draw k domain_ids using multinomial on probs
        raise NotImplementedError("Not implemented.")

    def update_acc(self, domain_id: str, passed_bool: bool):
        self.state.update_acc(domain_id, passed_bool)

    def state_dict(self) -> dict:
        return self.state.state_dict()
    def load_state_dict(self, sd: dict):
        self.state.load_state_dict(sd)