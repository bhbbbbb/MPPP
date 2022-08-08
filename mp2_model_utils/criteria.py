from model_utils.base.criteria import Loss as BaseLoss, Criteria


Loss = Criteria.register_criterion(
    short_name='loss',
    full_name='Loss',
    plot=True,
    primary=True,
)(BaseLoss)

SumationLoss = Criteria.register_criterion(
    short_name='sumation-loss',
    full_name='Sumation Loss',
    plot=True,
    primary=False,
)(BaseLoss)

DebutLoss = Criteria.register_criterion(
    short_name='debut-loss',
    full_name='Debut Loss',
    plot=True,
    primary=False,
)(BaseLoss)
