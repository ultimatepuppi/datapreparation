import awkward as ak
import numpy as np
import uproot as uproot


def smear_hcal(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_id,gen_pfs_charge,minpid):
    from awkward import Array
    from awkward.layout import ListOffsetArray64

    #Convert it to a 1D numpy array and perform smearing                                                                            
    numpy_e_arr = np.asarray(gen_pfs_e.layout.content)
    numpy_pt_arr = np.asarray(gen_pfs_pt.layout.content)
    numpy_eta_arr = np.asarray(gen_pfs_e.layout.content)
    numpy_id_arr = np.asarray(gen_pfs_id.layout.content)
    numpy_charge_arr = np.asarray(gen_pfs_charge.layout.content)


    smeared_e_arr = np.where((np.abs(numpy_eta_arr) <= 2.5), np.sqrt((numpy_e_arr**2)*(0.009**2) + numpy_e_arr*(0.12**2) + 0.45**2), numpy_e_arr)

    smeared_e_arr = np.where((np.abs(numpy_eta_arr) > 2.5), np.sqrt((numpy_e_arr**2)*(0.08**2) + numpy_e_arr*(1.98**2)), smeared_e_arr)

    # only apply smearing to desired particles matching pid 
    smeared_e_arr = np.where(np.abs(numpy_id_arr)>minpid,smeared_e_arr,numpy_e_arr)
    smeared_pt_arr = np.where((np.abs(numpy_id_arr)>minpid),numpy_e_arr/np.cosh(numpy_eta_arr),numpy_pt_arr)
    

    #Convert it back to awkward form
    return Array(ListOffsetArray64(gen_pfs_e.layout.offsets, Array(smeared_e_arr).layout)),Array(ListOffsetArray64(gen_pfs_pt.layout.offsets, Array(smeared_pt_arr).layout))


def smear_chargedhad(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_id,gen_pfs_charge,minpid):
    from awkward import Array
    from awkward.layout import ListOffsetArray64

    #Convert it to a 1D numpy array and perform smearing                                                                            
    numpy_e_arr = np.asarray(gen_pfs_e.layout.content)
    numpy_pt_arr = np.asarray(gen_pfs_pt.layout.content)
    numpy_eta_arr = np.asarray(gen_pfs_eta.layout.content)
    numpy_id_arr = np.asarray(gen_pfs_id.layout.content)
    numpy_charge_arr = np.asarray(gen_pfs_charge.layout.content)


    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.0000 ) & ( np.abs(numpy_eta_arr) < 0.2000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.00467469) ), numpy_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.0000 ) & ( np.abs(numpy_eta_arr) < 0.2000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.004675 + (numpy_pt_arr-1.000000)* 0.000056) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.0000 ) & ( np.abs(numpy_eta_arr) < 0.2000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.005179 + (numpy_pt_arr-10.000000)* 0.000064) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.0000 ) & ( np.abs(numpy_eta_arr) < 0.2000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.010955*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.2000 ) & ( np.abs(numpy_eta_arr) < 0.4000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.00515885) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.2000 ) & ( np.abs(numpy_eta_arr) < 0.4000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.005159 + (numpy_pt_arr-1.000000)* 0.000049) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.2000 ) & ( np.abs(numpy_eta_arr) < 0.4000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.005600 + (numpy_pt_arr-10.000000)* 0.000062) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.2000 ) & ( np.abs(numpy_eta_arr) < 0.4000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.011149*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.4000 ) & ( np.abs(numpy_eta_arr) < 0.6000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.00509775) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.4000 ) & ( np.abs(numpy_eta_arr) < 0.6000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.005098 + (numpy_pt_arr-1.000000)* 0.000055) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.4000 ) & ( np.abs(numpy_eta_arr) < 0.6000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.005590 + (numpy_pt_arr-10.000000)* 0.000063) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.4000 ) & ( np.abs(numpy_eta_arr) < 0.6000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.011251*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.6000 ) & ( np.abs(numpy_eta_arr) < 0.8000) ) & ( (numpy_pt_arr >= 0.0000 )& ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.00568785) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.6000 ) & ( np.abs(numpy_eta_arr) < 0.8000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.005688 + (numpy_pt_arr-1.000000)* 0.000061) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.6000 ) & ( np.abs(numpy_eta_arr) < 0.8000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.006237 + (numpy_pt_arr-10.000000)* 0.000059) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.6000 ) & ( np.abs(numpy_eta_arr) < 0.8000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.011563*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.8000 ) & ( np.abs(numpy_eta_arr) < 1.0000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.00668287) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.8000 ) & ( np.abs(numpy_eta_arr) < 1.0000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.006683 + (numpy_pt_arr-1.000000)* 0.000065) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.8000 ) & ( np.abs(numpy_eta_arr) < 1.0000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.007263 + (numpy_pt_arr-10.000000)* 0.000067) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 0.8000 ) & ( np.abs(numpy_eta_arr) < 1.0000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.013251*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.0000 ) & ( np.abs(numpy_eta_arr) < 1.2000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.01047734) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.0000 ) & ( np.abs(numpy_eta_arr) < 1.2000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.010477 + (numpy_pt_arr-1.000000)* 0.000059) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.0000 ) & ( np.abs(numpy_eta_arr) < 1.2000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.011005 + (numpy_pt_arr-10.000000)* 0.000098) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.0000 ) & ( np.abs(numpy_eta_arr) < 1.2000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.019785*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.2000 ) & ( np.abs(numpy_eta_arr) < 1.4000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.01430653) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.2000 ) & ( np.abs(numpy_eta_arr) < 1.4000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.014307 + (numpy_pt_arr-1.000000)* 0.000038) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.2000 ) & ( np.abs(numpy_eta_arr) < 1.4000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.014647 + (numpy_pt_arr-10.000000)* 0.000087) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.2000 ) & ( np.abs(numpy_eta_arr) < 1.4000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.022499*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.4000 ) & ( np.abs(numpy_eta_arr) < 1.6000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.01719240) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.4000 ) & ( np.abs(numpy_eta_arr) < 1.6000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.017192 + (numpy_pt_arr-1.000000)* 0.000020) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.4000 ) & ( np.abs(numpy_eta_arr) < 1.6000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.017372 + (numpy_pt_arr-10.000000)* 0.000080) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.4000 ) & ( np.abs(numpy_eta_arr) < 1.6000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.024604*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.6000 ) & ( np.abs(numpy_eta_arr) < 1.8000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.01783940) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.6000 ) & ( np.abs(numpy_eta_arr) < 1.8000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.017839 + (numpy_pt_arr-1.000000)* 0.000045) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.6000 ) & ( np.abs(numpy_eta_arr) < 1.8000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.018244 + (numpy_pt_arr-10.000000)* 0.000073) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.6000 ) & ( np.abs(numpy_eta_arr) < 1.8000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.024839*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.8000 ) & ( np.abs(numpy_eta_arr) < 2.0000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.01787758) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.8000 ) & ( np.abs(numpy_eta_arr) < 2.0000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.017878 + (numpy_pt_arr-1.000000)* 0.000154) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.8000 ) & ( np.abs(numpy_eta_arr) < 2.0000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.019265 + (numpy_pt_arr-10.000000)* 0.000141) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 1.8000 ) & ( np.abs(numpy_eta_arr) < 2.0000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.031979*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.0000 ) & ( np.abs(numpy_eta_arr) < 2.2000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.01825549) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.0000 ) & ( np.abs(numpy_eta_arr) < 2.2000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.018255 + (numpy_pt_arr-1.000000)* 0.000270) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.0000 ) & ( np.abs(numpy_eta_arr) < 2.2000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.020690 + (numpy_pt_arr-10.000000)* 0.000212) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.0000 ) & ( np.abs(numpy_eta_arr) < 2.2000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.039811*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.2000 ) & ( np.abs(numpy_eta_arr) < 2.4000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.01803308) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.2000 ) & ( np.abs(numpy_eta_arr) < 2.4000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.018033 + (numpy_pt_arr-1.000000)* 0.000220) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.2000 ) & ( np.abs(numpy_eta_arr) < 2.4000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.020013 + (numpy_pt_arr-10.000000)* 0.000247) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.2000 ) & ( np.abs(numpy_eta_arr) < 2.4000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.042262*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.4000 ) & ( np.abs(numpy_eta_arr) < 2.6000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.02156195) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.4000 ) & ( np.abs(numpy_eta_arr) < 2.6000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.021562 + (numpy_pt_arr-1.000000)* 0.000225) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.4000 ) & ( np.abs(numpy_eta_arr) < 2.6000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.023591 + (numpy_pt_arr-10.000000)* 0.000361) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.4000 ) & ( np.abs(numpy_eta_arr) < 2.6000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.056075*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.6000 ) & ( np.abs(numpy_eta_arr) < 2.8000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.02691276) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.6000 ) & ( np.abs(numpy_eta_arr) < 2.8000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.026913 + (numpy_pt_arr-1.000000)* 0.000357) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.6000 ) & ( np.abs(numpy_eta_arr) < 2.8000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.030127 + (numpy_pt_arr-10.000000)* 0.000483) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.6000 ) & ( np.abs(numpy_eta_arr) < 2.8000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.073578*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.8000 ) & ( np.abs(numpy_eta_arr) < 3.0000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.03253153) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.8000 ) & ( np.abs(numpy_eta_arr) < 3.0000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.032532 + (numpy_pt_arr-1.000000)* 0.000382) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.8000 ) & ( np.abs(numpy_eta_arr) < 3.0000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.035969 + (numpy_pt_arr-10.000000)* 0.000673) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 2.8000 ) & ( np.abs(numpy_eta_arr) < 3.0000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.096568*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.0000 ) & ( np.abs(numpy_eta_arr) < 3.2000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.04551714) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.0000 ) & ( np.abs(numpy_eta_arr) < 3.2000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.045517 + (numpy_pt_arr-1.000000)* 0.002188) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.0000 ) & ( np.abs(numpy_eta_arr) < 3.2000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.065207 + (numpy_pt_arr-10.000000)* 0.003254) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.0000 ) & ( np.abs(numpy_eta_arr) < 3.2000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.358089*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.2000 ) & ( np.abs(numpy_eta_arr) < 3.4000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.04724756) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.2000 ) & ( np.abs(numpy_eta_arr) < 3.4000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.047248 + (numpy_pt_arr-1.000000)* 0.003322) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.2000 ) & ( np.abs(numpy_eta_arr) < 3.4000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.077142 + (numpy_pt_arr-10.000000)* 0.004516) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.2000 ) & ( np.abs(numpy_eta_arr) < 3.4000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.483608*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.4000 ) & ( np.abs(numpy_eta_arr) < 3.6000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.04493020) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.4000 ) & ( np.abs(numpy_eta_arr) < 3.6000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.044930 + (numpy_pt_arr-1.000000)* 0.002917) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.4000 ) & ( np.abs(numpy_eta_arr) < 3.6000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.071183 + (numpy_pt_arr-10.000000)* 0.004186) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.4000 ) & ( np.abs(numpy_eta_arr) < 3.6000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.447915*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.6000 ) & ( np.abs(numpy_eta_arr) < 3.8000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.05464199) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.6000 ) & ( np.abs(numpy_eta_arr) < 3.8000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.054642 + (numpy_pt_arr-1.000000)* 0.003456) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.6000 ) & ( np.abs(numpy_eta_arr) < 3.8000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.085748 + (numpy_pt_arr-10.000000)* 0.004995) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.6000 ) & ( np.abs(numpy_eta_arr) < 3.8000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.535262*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.8000 ) & ( np.abs(numpy_eta_arr) < 4.0000) ) & ( (numpy_pt_arr >= 0.0000 ) & ( numpy_pt_arr < 1.0000) ) , np.random.normal(numpy_pt_arr, (0.06551185) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.8000 ) & ( np.abs(numpy_eta_arr) < 4.0000) ) & ( (numpy_pt_arr >= 1.0000 ) & ( numpy_pt_arr < 10.0000) ) , np.random.normal(numpy_pt_arr, (0.065512 + (numpy_pt_arr-1.000000)* 0.006574) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.8000 ) & ( np.abs(numpy_eta_arr) < 4.0000) ) & ( (numpy_pt_arr >= 10.0000 ) & ( numpy_pt_arr < 100.0000) ) , np.random.normal(numpy_pt_arr, (0.124674 + (numpy_pt_arr-10.000000)* 0.009560) ), smeared_pt_arr)
    smeared_pt_arr = np.where( ( (np.abs(numpy_eta_arr) >= 3.8000 ) & ( np.abs(numpy_eta_arr) < 4.0000) ) & ( (numpy_pt_arr >= 100.0000) ) , np.random.normal(numpy_pt_arr, (0.985040*numpy_pt_arr/100.000000) ), smeared_pt_arr)
    

    smeared_pt_arr = np.where((np.abs(numpy_id_arr)>minpid) & (numpy_charge_arr!=0),smeared_pt_arr,numpy_pt_arr)
    smeared_e_arr = np.where((np.abs(numpy_id_arr)>minpid) & (numpy_charge_arr!=0),np.cosh(numpy_eta_arr)*numpy_pt_arr,numpy_e_arr)

    #Convert it back to awkward form
    return Array(ListOffsetArray64(gen_pfs_e.layout.offsets, Array(smeared_e_arr).layout)),Array(ListOffsetArray64(gen_pfs_pt.layout.offsets, Array(smeared_pt_arr).layout))
