from abc import ABCMeta, abstractmethod

class BaseEnvironment:
    """
    Defines the interface of an RLGlue environment

    ie. These methods must be defined in your own environment classes
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, num_driver=30000, order_per_hour=0, on_offline=True, order_control=False):
        """Declare environment variables."""

    @abstractmethod
    def env_init(self):
        """
        Initialize environment variables.
        """

    @abstractmethod
    def env_start(self, num_online_drivers, num_orders, driver_dist):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

    @abstractmethod
    def generate_observation_od(self):
        """
        :return: generate observation of order dispatching
        """

    @abstractmethod
    def env_update_od(self, action):
        """
        Args:

        Returns:
        """

    @abstractmethod
    def generate_observation_rp(self):
        """
        :return: generate observation of reposition
        """

    @abstractmethod
    def env_update_rp(self, action):
        """
        Args:

        Returns:
        """

    @abstractmethod
    def env_update(self):
        """
        Update the state of the drivers

        :return:
        """



    @abstractmethod
    def env_message(self, message):
        """
        receive a message from RLGlue
        Args:
           message (str): the message passed
        Returns:
           str: the environment's response to the message (optional)
        """