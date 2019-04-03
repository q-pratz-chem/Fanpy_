"""Test wfns.wavefunction.cisd."""
import pytest
import numpy as np
from wfns.wfn.ci.cisd import CISD
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.solver.ci import brute
from utils import skip_init, find_datafile


def test_cisd_assign_sd_vec():
    """Test CISD.assign_sd_vec."""
    test = skip_init(CISD)
    test.assign_nelec(3)
    test.assign_nspin(6)
    test.assign_spin(None)
    test.assign_seniority(None)
    test.assign_sd_vec()
    assert test.sd_vec == (
        0b001011,
        0b011001,
        0b001101,
        0b101001,
        0b011010,
        0b001110,
        0b101010,
        0b010011,
        0b000111,
        0b100011,
        0b011100,
        0b111000,
        0b101100,
        0b010101,
        0b110001,
        0b100101,
        0b010110,
        0b110010,
        0b100110,
    )
    with pytest.raises(ValueError):
        test.assign_sd_vec((0b001011, 0b011001))


def test_cisd_h2_631gdp():
    """Test CISD wavefunction using H2 (6-31G**).

    Compared to Gausssian results
    HF energy: -1.13126983927
    FCI energy: -1.1651487496

    Note
    ----
    You have to be careful with Gaussian and molpro calculations because they freeze core by
    default
    """
    nelec = 2
    nspin = 20
    cisd = CISD(nelec, nspin)

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    nuc_nuc = 0.71317683129
    ham = RestrictedChemicalHamiltonian(one_int, two_int, energy_nuc_nuc=nuc_nuc)

    # optimize
    results = brute(cisd, ham)
    energy = results["energy"]
    # compare with number from Gaussian
    assert abs(energy + nuc_nuc - (-1.1651486697)) < 1e-7


def test_cisd_lih_631g():
    """Test CISD wavefunction using LiH (6-31G).

    Compared to Molpro results
    HF energy: -7.97926895
    CISD energy: -7.99826182

    Note
    ----
    You have to be careful with Gaussian and molpro calculations because they freeze core by
    default
    """
    nelec = 4
    nspin = 22
    cisd = CISD(nelec, nspin)

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_631g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data_lih_hf_631g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_631g_twoint.npy"))
    nuc_nuc = 0.995317634356
    ham = RestrictedChemicalHamiltonian(one_int, two_int, energy_nuc_nuc=nuc_nuc)

    # optimize
    results = brute(cisd, ham)
    energy = results["energy"]
    # compare with number from Gaussian
    assert abs(energy + nuc_nuc - (-7.99826182)) < 1e-7
