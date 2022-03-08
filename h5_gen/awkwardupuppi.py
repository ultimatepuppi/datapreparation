import h5py
import sys
import awkward as ak
import numpy as np
import uproot as uproot
import smear_chargedhadrons as sch

#np.random.seed(0)

file = uproot.open(sys.argv[1])

events = file['Delphes']
nEvents = 1000

# read branches
def read_branch(branchname):
    return events[branchname].array(entry_stop=nEvents)

gen_vtx = read_branch('GenVertex_size')
gen_vtx_x = read_branch('GenVertex.X')
gen_vtx_y = read_branch('GenVertex.Y')
gen_vtx_z = read_branch('GenVertex.Z')
gen_pfs_pt = read_branch('PileUpMix.PT')
gen_pfs_eta = read_branch('PileUpMix.Eta')
gen_pfs_phi = read_branch('PileUpMix.Phi')
gen_pfs_e = read_branch('PileUpMix.E')
gen_pfs_pid = read_branch('PileUpMix.PID')
gen_pfs_charge = read_branch('PileUpMix.Charge')
gen_pfs_genvtx = read_branch('PileUpMix.GenVtxIdx')
gen_pfs_genvtx = gen_pfs_genvtx + 1

#print(gen_pfs_pt)



# neutrino mask
neutrino_mask = (np.abs(gen_pfs_pid) != 12) & (np.abs(gen_pfs_pid) != 14) & (np.abs(gen_pfs_pid) != 16)
def filter_neutrinos(arr):
    newarr = arr[neutrino_mask]
    return newarr

gen_pfs_pt = filter_neutrinos(gen_pfs_pt)
#print(gen_pfs_pt)

gen_pfs_eta = filter_neutrinos(gen_pfs_eta)
gen_pfs_phi = filter_neutrinos(gen_pfs_phi)
gen_pfs_e = filter_neutrinos(gen_pfs_e)
gen_pfs_pid = filter_neutrinos(gen_pfs_pid)
gen_pfs_charge = filter_neutrinos(gen_pfs_charge)
gen_pfs_genvtx = filter_neutrinos(gen_pfs_genvtx)

# Smearing photon energies
def smear_em(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_id,pid):
    from awkward import Array
    from awkward.layout import ListOffsetArray64
    
    #Convert it to a 1D numpy array and perform smearing
    numpy_e_arr = np.asarray(gen_pfs_e.layout.content)
    numpy_pt_arr = np.asarray(gen_pfs_pt.layout.content)
    numpy_eta_arr = np.asarray(gen_pfs_e.layout.content)
    numpy_id_arr = np.asarray(gen_pfs_id.layout.content)

    if pid == 22:
        smeared_e_arr = np.random.normal(numpy_e_arr, numpy_e_arr*0.01)
    elif pid == 11:
        smeared_e_arr = np.where(np.abs(numpy_eta_arr)<1.5,
                                 np.random.normal(numpy_e_arr, numpy_e_arr*0.028),
                                 numpy_e_arr)
        smeared_e_arr = np.where(((np.abs(numpy_eta_arr)>1.5) & (np.abs(numpy_eta_arr)<=1.75)),
                                 np.random.normal(numpy_e_arr, numpy_e_arr*0.037),
                                 smeared_e_arr)
        smeared_e_arr = np.where(((np.abs(numpy_eta_arr)>1.75) & (np.abs(numpy_eta_arr)<=2.15)),
                                 np.random.normal(numpy_e_arr, numpy_e_arr*0.038),
                                 smeared_e_arr)
        smeared_e_arr = np.where(((np.abs(numpy_eta_arr)>2.15) & (np.abs(numpy_eta_arr)<=3.00)),
                                 np.random.normal(numpy_e_arr, numpy_e_arr*0.044),
                                 smeared_e_arr)
        smeared_e_arr = np.where(((np.abs(numpy_eta_arr)>3.00) & (np.abs(numpy_eta_arr)<=4.00)),
                                 np.random.normal(numpy_e_arr, numpy_e_arr*0.10),
                                 smeared_e_arr)

    # only apply smearing to desired particles matching pid
    smeared_e_arr = np.where(np.abs(numpy_id_arr)==pid,smeared_e_arr,numpy_e_arr)
    smeared_pt_arr = np.where(np.abs(numpy_id_arr)==pid,smeared_e_arr/np.cosh(numpy_eta_arr),numpy_pt_arr)

    #Convert it back to awkward form
    return Array(ListOffsetArray64(gen_pfs_e.layout.offsets, Array(smeared_e_arr).layout)),Array(ListOffsetArray64(gen_pfs_pt.layout.offsets, Array(smeared_pt_arr).layout))

#print(gen_pfs_pt)

gen_pfs_e, gen_pfs_pt = smear_em(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_pid,22)
gen_pfs_e, gen_pfs_pt = smear_em(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_pid,11)
gen_pfs_e, gen_pfs_pt = sch.smear_chargedhad(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_pid,gen_pfs_charge,22)

#print(gen_pfs_pt)

#gen_pfs_e, gen_pfs_pt = sch.smear_hcal(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_pid,gen_pfs_charge,22)

def one_hot_pid(gen_pfs_pid,gen_pfs_charge,gen_pfs_eta):
    from awkward import Array
    from awkward.layout import ListOffsetArray64
    
    #Convert it to a 1D numpy array
    numpy_pid_arr = np.asarray(gen_pfs_pid.layout.content)
    numpy_charge_arr = np.asarray(gen_pfs_charge.layout.content)
    numpy_eta_arr = np.asarray(gen_pfs_eta.layout.content)

    # Convert electrons outside tracker to photons
    new_pid_arr = np.where(((np.abs(numpy_eta_arr)>3.0) & (np.abs(numpy_pid_arr)==11)),22.,numpy_pid_arr)
    # Convert charged hadrons outside tracker to neutral hadrons (0)
    new_pid_arr = np.where(((np.abs(numpy_eta_arr)>3.0) & (numpy_charge_arr!=0) & (np.abs(numpy_pid_arr)>13)),0.,new_pid_arr)
    # Convert all neutral hadrons to default 0
    new_pid_arr = np.where(((numpy_charge_arr==0) & ( np.abs(numpy_pid_arr)>22)),0.,new_pid_arr)
    # Convert all charged hadrons within tracker to pion or non-pion
    new_pid_arr = np.where(((np.abs(numpy_eta_arr)<3.0) & (numpy_charge_arr!=0) & (np.abs(numpy_pid_arr)!=211)),1.,new_pid_arr)
    new_pid_arr = np.where(((np.abs(numpy_eta_arr)<3.0) & (numpy_charge_arr!=0) & (np.abs(numpy_pid_arr)==211)),2.,new_pid_arr)
    # Give electrons inside tracker to 3
    new_pid_arr = np.where(((np.abs(numpy_eta_arr)<3.0) & (np.abs(numpy_pid_arr)==11)),3.,new_pid_arr)
    # Give photons 4
    new_pid_arr = np.where((np.abs(new_pid_arr)==22),4.,new_pid_arr)

    return Array(ListOffsetArray64(gen_pfs_pid.layout.offsets, Array(new_pid_arr).layout))

new_gen_pfs_pid = one_hot_pid(gen_pfs_pid,gen_pfs_charge,gen_pfs_eta)


def disable_charge(gen_pfs_charge,gen_pfs_eta,gen_pfs_genvtx):
    from awkward import Array
    from awkward.layout import ListOffsetArray64
    
    #Convert it to a 1D numpy array
    numpy_charge_arr = np.asarray(gen_pfs_charge.layout.content)
    numpy_eta_arr = np.asarray(gen_pfs_eta.layout.content)
    numpy_genvtx_arr = np.asarray(gen_pfs_genvtx.layout.content)

    new_charge_arr = np.where(numpy_eta_arr<3.0,numpy_charge_arr,0.)
    new_vtx_arr = np.where(numpy_charge_arr==0,-1,numpy_genvtx_arr)
    return Array(ListOffsetArray64(gen_pfs_charge.layout.offsets, Array(new_charge_arr).layout)),  Array(ListOffsetArray64(gen_pfs_genvtx.layout.offsets, Array(new_vtx_arr).layout))


gen_pfs_charge, gen_pfs_vtx = disable_charge(gen_pfs_charge,gen_pfs_eta,gen_pfs_genvtx)

# kinematics mask
kinematics_mask = (gen_pfs_pt>1.0) & (np.abs(gen_pfs_eta)<5.)
def filter_minpt(arr):
    newarr = arr[kinematics_mask]
    return newarr

gen_pfs_pt = filter_minpt(gen_pfs_pt)

gen_pfs_eta = filter_minpt(gen_pfs_eta)
gen_pfs_phi = filter_minpt(gen_pfs_phi)
gen_pfs_e = filter_minpt(gen_pfs_e)
gen_pfs_pid = filter_minpt(gen_pfs_pid)
gen_pfs_charge = filter_minpt(gen_pfs_charge)
gen_pfs_vtx = filter_minpt(gen_pfs_vtx)
gen_pfs_genvtx = filter_minpt(gen_pfs_genvtx)



gen_vtx_id = ak.local_index(gen_vtx_z)
#print(gen_vtx_id)


def fill_ndf(gen_pfs_vtx,gen_pfs_charge,gen_vtx_id):
    from awkward import Array
    from awkward.layout import ListOffsetArray64

    # filter neutrals
    gen_pfs_vtx = gen_pfs_vtx[gen_pfs_charge != 0]


    big_vtxcounts = []
    # event loop
    for e in range(len(gen_pfs_vtx)):
        vtxcounts = []
        
        #Convert it to a 1D numpy array
        numpy_vtx_arr = np.asarray(gen_pfs_vtx[e])

        # gen vtx loop
        for v in gen_vtx_id[e]:
            vtxcounts.append((numpy_vtx_arr == v).sum())


        big_vtxcounts.append(vtxcounts)
    #print(big_vtxcounts)
    
    
    return ak.Array(big_vtxcounts)


gen_vtx_ndf = fill_ndf(gen_pfs_vtx,gen_pfs_charge,gen_vtx_id)

def from_good_vertex(gen_pfs_genvtx,gen_pfs_charge,gen_vtx_ndf):
    
    big_gen_pfs_genvtx = []
    
    for e in range(len(gen_pfs_genvtx)):

        tmp_gen_pfs_genvtx = []

        for p in range(len(gen_pfs_genvtx[e])):
            if gen_pfs_charge[e][p] == 0 and gen_vtx_ndf[e][gen_pfs_genvtx[e][p]] == 0:
                tmp_gen_pfs_genvtx.append(-1.)
            else:
                tmp_gen_pfs_genvtx.append(gen_pfs_genvtx[e][p])
                
        big_gen_pfs_genvtx.append(tmp_gen_pfs_genvtx)

    return ak.Array(big_gen_pfs_genvtx)


gen_pfs_genvtx = from_good_vertex(gen_pfs_genvtx,gen_pfs_charge,gen_vtx_ndf)
#print(gen_pfs_genvtx)


gen_vtx_ndf = gen_vtx_ndf[gen_vtx_ndf>0]
gen_vtx_x = gen_vtx_x[gen_vtx_ndf>0]
gen_vtx_y = gen_vtx_y[gen_vtx_ndf>0]
gen_vtx_z = gen_vtx_z[gen_vtx_ndf>0]
gen_vtx_id = gen_vtx_id[gen_vtx_ndf>0]

#exit(1)



final_pfs_pt = np.array(ak.fill_none(ak.pad_none(gen_pfs_pt,7000,clip=True),-99,axis=-1))
final_pfs_eta = np.array(ak.fill_none(ak.pad_none(gen_pfs_eta,7000,clip=True),-99,axis=-1))
final_pfs_phi = np.array(ak.fill_none(ak.pad_none(gen_pfs_phi,7000,clip=True),-99,axis=-1))
final_pfs_e = np.array(ak.fill_none(ak.pad_none(gen_pfs_e,7000,clip=True),-99,axis=-1))
final_pfs_pid = np.array(ak.fill_none(ak.pad_none(gen_pfs_pid,7000,clip=True),-99,axis=-1))
final_pfs_charge = np.array(ak.fill_none(ak.pad_none(gen_pfs_pid,7000,clip=True),-99,axis=-1))
final_pfs_vtx = np.array(ak.fill_none(ak.pad_none(gen_pfs_vtx,7000,clip=True),-99,axis=-1))
final_pfs_genvtx = np.array(ak.fill_none(ak.pad_none(gen_pfs_genvtx,7000,clip=True),-99,axis=-1))

final_pfs_features = np.stack((final_pfs_pt,final_pfs_eta,final_pfs_phi,final_pfs_e,final_pfs_pid,final_pfs_charge),axis=-1)

#print(final_pfs_vtx)
#print(final_pfs_vtx.shape)
print(final_pfs_features)
print(final_pfs_features.shape)


final_vtx_x = np.array(ak.fill_none(ak.pad_none(gen_vtx_x,200,clip=True),-99,axis=-1))
final_vtx_y = np.array(ak.fill_none(ak.pad_none(gen_vtx_y,200,clip=True),-99,axis=-1))
final_vtx_z = np.array(ak.fill_none(ak.pad_none(gen_vtx_z,200,clip=True),-99,axis=-1))
final_vtx_ndf = np.array(ak.fill_none(ak.pad_none(gen_vtx_ndf,200,clip=True),-99,axis=-1))
final_vtx_id = np.array(ak.fill_none(ak.pad_none(gen_vtx_id,200,clip=True),-99,axis=-1))

final_vtx_features = np.stack((final_vtx_x,final_vtx_y,final_vtx_z,final_vtx_ndf),axis=-1)

hf = h5py.File(sys.argv[2], 'w')

hf.create_dataset('pfs', data=final_pfs_features)
hf.create_dataset('pfs_shape', data=final_pfs_features.shape)
hf.create_dataset('edge_start', data=final_pfs_vtx)
hf.create_dataset('edge_start_shape', data=final_pfs_vtx.shape)
hf.create_dataset('edge_stop', data=final_vtx_id)
hf.create_dataset('edge_stop_shape', data=final_vtx_id.shape)
hf.create_dataset('vtx', data=final_vtx_features)
hf.create_dataset('vtx_shape', data=final_vtx_features.shape)
hf.create_dataset('truth', data=final_pfs_genvtx)
hf.create_dataset('truth_shape', data=final_pfs_genvtx.shape)
hf.create_dataset('n', data=final_pfs_genvtx.shape[0])

hf.close()

