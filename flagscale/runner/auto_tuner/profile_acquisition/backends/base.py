from abc import ABC, abstractmethod


class ProfileBackend(ABC):
    @abstractmethod
    def collect_device_memory(self):
        raise NotImplementedError

    @abstractmethod
    def collect_collectives(self, p2p_command, all_reduce_command, **kwargs):
        raise NotImplementedError
