"""Seniority-zero Hamiltonian object that interacts with the wavefunction."""
import numpy as np
from wfns.backend import slater
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian


class SeniorityZeroHamiltonian(RestrictedChemicalHamiltonian):
    r"""Hamiltonian that involves only the zero-seniority terms.

    # FIXME: fix up eqns
    .. math::

        \def\i{\imath}
        \def\j{\jmath}
        \hat{H} =
        \sum_{i} \left(
            h_{ii} a^\dagger_i a_i + h_{\bar{\i}\bar{\i}} a^\dagger_{\bar{\i}} a_{\bar{\i}}
        \right)
        + \sum_{i<j} \left(
            g_{ijij} a^\dagger_i a^\dagger_j a_j a_i
            + g_{i\bar{\j}i\bar{\j}} a^\dagger_i a^\dagger_{\bar{\j}} a_{\bar{\j}} a_i
            + g_{\bar{\i}j\bar{\i}j} a^\dagger_{\bar{\i}} a^\dagger_j  a_j a_{\bar{\i}}
            + g_{\bar{\i}\bar{\j}\bar{\i}\bar{\j}}
            a^\dagger_{\bar{\i}} a^\dagger_{\bar{\j}} a_{\bar{\j}} a_{\bar{\i}}
            + g_{i\bar{\j}\bar{\j}i} a^\dagger_i a^\dagger_{\bar{\j}} a_{\bar{\j} a_i}
        \right)

    where :math:`i` and :math:`\bar{\i}` are the indices that correspond to the ith alpha- and beta-
    spin orbtials, :math:`h_{ik}` is the one-electron integral and :math:`g_{ijkl}` is the two-
    electron integral.

    Seniority zero means that there are no unpaired electrons.

    Attributes
    ----------
    params : np.ndarray
        Significant elements of the anti-Hermitian matrix.
    energy_nuc_nuc : float
        Nuclear-nuclear repulsion energy.
    one_int : {1- or 2-tuple np.ndarray(K, K)}
        One electron integrals for restricted, unrestricted, or generalized orbitals.
        1-tuple for spatial (restricted) and generalized orbitals.
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components).
    two_int : {1- or 3-tuple np.ndarray(K, K)}
        Two electron integrals for restricted, unrestricted, or generalized orbitals.
        Uses the physicist's notation.
        1-tuple for spatial (restricted) and generalized orbitals.
        3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components).

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspin : int
        Number of spin orbitals.
    nspatial : int
        Number of spatial orbitals.

    Methods
    -------
    __init__(self, one_int, two_int, orbtype=None, energy_nuc_nuc=None)
        Initialize the Hamiltonian
    clear_cache(self)
        Placeholder function that would clear the cache.
    assign_orbtype(self, orbtype=None)
        Assign the orbital type.
    assign_energy_nuc_nuc(self, energy_nuc_nuc=None)
        Assigns the nuclear nuclear repulsion.
    assign_integrals(self, one_int, two_int)
        Assign the one- and two-electron integrals.
    assign_params(self, params)
        Transform the integrals with a unitary matrix that corresponds to the given parameters.
    orb_rotate_jacobi(self, jacobi_indices, theta)
        Rotate orbitals using Jacobi matrix.
    orb_rotate_matrix(self, matrix)
        Rotate orbitals using a transformation matrix.
    integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """

    def integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None):
        r"""Integrate the Hamiltonian with against a wavefunction and Slater determinant.

        .. math::

            \left< \Phi \middle| \hat{H} \middle| \Psi \right>
            = \sum_{\mathbf{m} \in S_\Phi}
              f(\mathbf{m}) \left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>

        where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
        :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
        determinants for which :math:`\left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>` is
        not zero, which are the :math:`\Phi` and its first order orbital-pair excitations for a
        chemical Hamiltonian.

        """
        if wfn_deriv is not None and ham_deriv is not None:
            raise ValueError(
                "Integral can be derivatized with respect to at most one out of the "
                "wavefunction and Hamiltonian parameters."
            )

        nspatial = self.nspin // 2
        sd = slater.internal_sd(sd)
        if slater.get_seniority(sd, nspatial) != 0:
            return 0.0, 0.0, 0.0

        sd_spatial = slater.split_spin(sd, nspatial)[0]
        occ_spatial_indices = slater.occ_indices(sd_spatial)
        vir_spatial_indices = slater.vir_indices(sd_spatial, nspatial)

        one_electron, coulomb, exchange = 0.0, 0.0, 0.0

        def update_integrals(sd_m):
            """Update the integral values."""
            coeff = wfn.get_overlap(sd_m, deriv=wfn_deriv)
            sd_energy = self.integrate_sd_sd(sd, sd_m, deriv=ham_deriv)
            return (
                one_electron + coeff * sd_energy[0],
                coulomb + coeff * sd_energy[1],
                exchange + coeff * sd_energy[2],
            )

        one_electron, coulomb, exchange = update_integrals(sd)

        for i in occ_spatial_indices:
            for a in vir_spatial_indices:  # pylint: disable=C0103
                sd_m = slater.excite(
                    sd,
                    slater.spin_index(i, nspatial, "alpha"),
                    slater.spin_index(i, nspatial, "beta"),
                    slater.spin_index(a, nspatial, "beta"),
                    slater.spin_index(a, nspatial, "alpha"),
                )
                one_electron, coulomb, exchange = update_integrals(sd_m)

        return one_electron, coulomb, exchange

    def integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None):
        r"""Integrate the Hamiltonian with against two Slater determinants.

        .. math::

            H_{\mathbf{m}\mathbf{n}} =
            \left< \mathbf{m} \middle| \hat{H} \middle| \mathbf{n} \right>

        In the first summation involving :math:`h_{ij}`, only the terms where :math:`\mathbf{m}` and
        :math:`\mathbf{n}` are the same will contribute to the integral. In the second summation
        involving :math:`g_{ijkl}`, only the terms where :math:`\mathbf{m}` and :math:`\mathbf{n}`
        are different by at most single pair-wise excitation will contribute to the integral.

        Parameters
        ----------
        sd1 : int
            Seniority-zero Slater Determinant.
        sd2 : int
            Seniority-zero Slater Determinant.
        sign : {1, -1, None}
            Sign change resulting from cancelling out the orbitals shared between the two Slater
            determinants.
            Computes the sign if none is provided.
            Make sure that the provided sign is correct. It will not be checked to see if its
            correct.
        deriv : {int, None}
            Index of the Hamiltonian parameter against which the integral is derivatized.
            Default is no derivatization.

        Returns
        -------
        one_electron : float
            One-electron energy.
        coulomb : float
            Coulomb energy.
        exchange : float
            Exchange energy.

        Raises
        ------
        ValueError
            If `sign` is not `1`, `-1` or `None`.
        NotImplementedError
            If `deriv` is not `None`.

        """
        # pylint: disable=C0103
        if deriv is not None:
            raise NotImplementedError(
                "Orbital rotation is not implemented properly: you cannot "
                "take the derivative of CI matrix elements with respect to "
                "orbital rotation coefficients."
            )

        nspatial = self.one_int[0].shape[0]

        sd1 = slater.internal_sd(sd1)
        sd2 = slater.internal_sd(sd2)

        # if the Slater determinants are not seniority zero
        if not slater.get_seniority(sd1, nspatial) == slater.get_seniority(sd2, nspatial) == 0:
            return 0.0, 0.0, 0.0

        sd1_spatial = slater.split_spin(sd1, nspatial)[0]
        sd2_spatial = slater.split_spin(sd2, nspatial)[0]
        shared_indices = np.array(slater.shared_orbs(sd1_spatial, sd2_spatial))
        diff_sd1, diff_sd2 = slater.diff_orbs(sd1_spatial, sd2_spatial)

        # if two Slater determinants do not have the same number of electrons
        if len(diff_sd1) != len(diff_sd2):
            return 0.0, 0.0, 0.0

        diff_order = len(diff_sd1)
        # if two Slater determinants are greater than double (spatial orbital) excitation away
        if diff_order > 1:
            return 0.0, 0.0, 0.0

        if sign is None:
            sign = 1
        elif sign not in [1, -1]:
            raise ValueError("The sign associated with the integral must be either `1` or `-1`.")

        one_electron, coulomb, exchange = 0.0, 0.0, 0.0
        # two sd's are the same
        if diff_order == 0:
            one_electron = 2 * np.sum(self.one_int[shared_indices, shared_indices])
            coulomb = 2 * np.sum(
                np.triu(self._cached_two_int_ijij[shared_indices[:, None], shared_indices], k=1)
            )
            coulomb += np.sum(self._cached_two_int_ijij[shared_indices[:, None], shared_indices])
            exchange = -2 * np.sum(
                np.triu(self._cached_two_int_ijji[shared_indices[:, None], shared_indices], k=1)
            )
        # two sd's are different by double excitation
        else:
            a, = diff_sd1
            b, = diff_sd2
            spatial_a = slater.spatial_index(a, nspatial)
            spatial_b = slater.spatial_index(b, nspatial)

            coulomb = self.two_int[spatial_a, spatial_a, spatial_b, spatial_b]

        return sign * one_electron, sign * coulomb, sign * exchange
