from .energy_oneside import EnergyOneSideProjection


class EnergyOneSideProjectionNoNorm(EnergyOneSideProjection):
    def objective(self, params, assign=True, normalize=False, save=True):
        # always disable normalization during optimization
        return super().objective(params, assign=assign, normalize=False, save=save)

    def gradient(self, params, assign=False, normalize=False, save=True):
        # also disable normalization for gradient calls
        return super().gradient(params, assign=assign, normalize=False, save=save)

