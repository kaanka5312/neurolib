from . import loadDefaultParams as dp
from . import timeIntegration as ti
from . import timeIntegration_edited as ti_edited

from ..model import Model


class WCModel(Model):
    """
    The two-population Wilson-Cowan model
    """

    name = "wc"
    description = "Wilson-Cowan model"

    init_vars = ["exc_init", "inh_init", "exc_ou", "inh_ou"]
    state_vars = ["exc", "inh", "exc_ou", "inh_ou"]
    output_vars = ["exc", "inh"]
    default_output = "exc"
    input_vars = ["exc_ext", "inh_ext"]
    default_input = "exc_ext"

    # because this is not a rate model, the input
    # to the bold model must be transformed
    boldInputTransform = lambda self, x: x * 50

    def __init__(self, params=None, Cmat=None, Dmat=None, seed=None, integration_type="original"):
        """
        Initialize the WCModel.

        Parameters:
            params (dict, optional): Model parameters.
            Cmat (array, optional): Connectivity matrix.
            Dmat (array, optional): Delay matrix.
            seed (int, optional): Random seed.
            integration_type (str, optional): Choose integration method. 
                                              "original" for timeIntegration or 
                                              "edited" for timeIntegration_edited.
        """
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed

        # Choose integration function based on integration_type
        if integration_type == "original":
            integration = ti.timeIntegration
        elif integration_type == "edited":
            integration = ti_edited.timeIntegration
        else:
            raise ValueError(f"Unknown integration_type: {integration_type}. Choose 'original' or 'edited'.")

        # Load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)

        # Initialize base class Model
        super().__init__(integration=integration, params=params)
