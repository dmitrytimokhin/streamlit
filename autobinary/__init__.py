
from autobinary.libraries import BackwardDifferenceEncoder, BinaryEncoder, CountEncoder
from autobinary.libraries.category_encoders import HashingEncoder, HelmertEncoder, OneHotEncoder
from autobinary.libraries.category_encoders import OrdinalEncoder, SumEncoder, PolynomialEncoder
from autobinary.libraries.category_encoders import BaseNEncoder, LeaveOneOutEncoder, TargetEncoder
from autobinary.libraries.category_encoders import WOEEncoder, MEstimateEncoder, JamesSteinEncoder
from autobinary.libraries.category_encoders import CatBoostEncoder, GLMMEncoder

from autobinary.libraries import SoloModel, ClassTransformation, TwoModels
from autobinary.libraries import TwoModelsExtra

from autobinary.libraries import (plot_uplift_preds, plot_qini_curve, 
        plot_uplift_curve, plot_uplift_by_percentile, 
        plot_treatment_balance_curve)

from autobinary.libraries import (
    uplift_curve, perfect_uplift_curve, uplift_auc_score,
    qini_curve, perfect_qini_curve, qini_auc_score,
    uplift_at_k, response_rate_by_percentile,
    weighted_average_uplift, uplift_by_percentile, treatment_balance_curve,
    average_squared_deviation, make_uplift_scorer
)

from autobinary.other_methods import StratifiedGroupKFold

from autobinary.trees import AutoTrees
from autobinary.trees import PermutationSelection, TargetPermutationSelection
from autobinary.trees import AutoSelection
from autobinary.trees import PlotShap, PlotPDP

from autobinary.utils import get_img_tag, df_html
from autobinary.utils import settings_style
from autobinary.utils import utils_target_permutation 

from autobinary.custom_metrics import BalanceCover

from autobinary.pipe import SentColumns, base_pipe

from autobinary.wing import WingOfEvidence, WingsOfEvidence

from autobinary.uplift_utils import get_uplift

from autobinary.auto_viz import TargetPlot

from autobinary.uplift_utils import UpliftCalibration

from autobinary.uplift_utils import WingUplift

__version__ = '1.0.5'

__author__ = '''
    vasily_sizov, 
    dmitrii_timohin, 
    pavel_pelenskii, 
    ruslan_popov'''
