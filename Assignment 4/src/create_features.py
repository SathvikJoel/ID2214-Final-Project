import numpy as np
import pandas as pd
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as d 
import rdkit.Chem.Fragments as f
import rdkit.Chem.Lipinski as l
from rdkit.Chem import AllChem

from tqdm import tqdm
# Create features
def MolFromSmiles(smile):
    try:
        return Chem.MolFromSmiles(smile)
    except:
        return np.nan

def GetNumAtoms(mol) -> int:
    try:
        return mol.GetNumAtoms()
    except:
        return np.nan

def ExactMolWt(mol) ->float:
    try:
        return d.CalcExactMolWt(mol)
    except:
        return np.nan

def CalcAsphericity(mol) -> float:
    try:
        return d.CalcAsphericity(mol)
    except:
        return np.nan

def CalcChi0n(mol) -> float:
    try:
        return d.CalcChi0n(mol)
    except:
        return np.nan

def CalcChi0v(mol) -> float:
    try:
        return d.CalcChi0v(mol)
    except:
        return np.nan

def CalcChi1n(mol) -> float:
    try:
        return d.CalcChi1n(mol)
    except:
        return np.nan

def CalcChi1v(mol) -> float:
    try:
        return d.CalcChi1v(mol)
    except:
        return np.nan

def CalcChi2n(mol) -> float:
    try:
        return d.CalcChi2n(mol)
    except:
        return np.nan

def CalcChi2v(mol) -> float:
    try:
        return d.CalcChi2v(mol)
    except:
        return np.nan

def CalcChi3n(mol) -> float:
    try:
        return d.CalcChi3n(mol)
    except:
        return np.nan

def CalcChi3v(mol) -> float:
    try:
        return d.CalcChi3v(mol)
    except:
        return np.nan


def CalcChi4n(mol) -> float:
    try:
        return d.CalcChi4n(mol)
    except:
        return np.nan


def CalcChi4v(mol) -> float:
    try:
        return d.CalcChi4v(mol)
    except:
        return np.nan


def CalcEccentricity(mol) -> float:
    try:
        return d.CalcEccentricity(mol)
    except:
        return np.nan


def CalcFractionCSP3(mol) -> float:
    try:
        return d.CalcFractionCSP3(mol)
    except:
        return np.nan


def CalcHallKierAlpha(mol) -> float:
    try:
        return d.CalcHallKierAlpha(mol)
    except:
        return np.nan

def CalcInertialShapeFactor(mol) -> float:
    try:
        return d.CalcInertialShapeFactor(mol)
    except:
        return np.nan

def CalcKappa1(mol) -> float:
    try:
        return d.CalcKappa1(mol)
    except:
        return np.nan

def CalcKappa2(mol) -> float:
    try:
        return d.CalcKappa2(mol)
    except:
        return np.nan

def CalcKappa3(mol) -> float:
    try:
        return d.CalcKappa3(mol)
    except:
        return np.nan

def CalcLabuteASA(mol) -> float:
    try:
        return d.CalcLabuteASA(mol)
    except:
        return np.nan


def CalcMORSE(mol) -> float:
    try:
        return d.CalcMORSE(mol)
    except:
        return np.nan

def CalcMolFormula(mol) -> str:
    try:
        return d.CalcMolFormula(mol)
    except:
        return np.nan

def CalcNPR1(mol) -> float:
    try:
        return d.CalcNPR1(mol)
    except:
        return np.nan

def CalcNPR2(mol) -> float:
    try:
        return d.CalcNPR2(mol)
    except:
        return np.nan


def CalcNumAliphaticCarbocycles(mol) -> int:
    try:
        return d.CalcNumAliphaticCarbocycles(mol)
    except:
        return np.nan

def CalcNumAliphaticHeterocycles(mol) -> int:
    try:
        return d.CalcNumAliphaticHeterocycles(mol)
    except:
        return np.nan

def CalcNumAliphaticRings(mol) -> int:
    try:
        return d.CalcNumAliphaticRings(mol)
    except:
        return np.nan

def CalcNumAmideBonds(mol) -> int:
    try:
        return d.CalcNumAmideBonds(mol)
    except:
        return np.nan

def CalcNumAromaticCarbocycles(mol) -> int:
    try:
        return d.CalcNumAromaticCarbocycles(mol)
    except:
        return np.nan

def CalcNumAromaticHeterocycles(mol) -> int:
    try:
        return d.CalcNumAromaticHeterocycles(mol)
    except: 
        return np.nan

def CalcNumAromaticRings(mol) -> int:
    try:
        return d.CalcNumAromaticRings(mol)
    except:
        return np.nan
def CalcNumAtomStereoCenters(mol) -> int:
    try:
        return d.CalcNumAtomStereoCenters(mol)
    except:
        return np.nan

def CalcNumAtoms(mol) -> int:
    try:
        return d.CalcNumAtoms(mol)
    except:
        return np.nan

def CalcNumBridgeheadAtoms(mol) -> int:
    try:
        return d.CalcNumBridgeheadAtoms(mol)
    except:
        return np.nan

def CalcNumHBA(mol) -> int:
    try:
        return d.CalcNumHBA(mol)
    except:
        return np.nan

def CalcNumHBD(mol) -> int:
    try:
        return d.CalcNumHBD(mol)
    except:
        return np.nan


def CalcNumHeavyAtoms(mol) -> int:
    try:
        return d.CalcNumHeavyAtoms(mol)
    except:
        return np.nan

def CalcNumHeteroatoms(mol) -> int:
    try:
        return d.CalcNumHeteroatoms(mol)
    except:
        return np.nan


def CalcNumHeterocycles(mol) -> int:
    try:
        return d.CalcNumHeterocycles(mol)
    except:
        return np.nan


def CalcNumLipinskiHBA(mol) -> int:
    try:
        return d.CalcNumLipinskiHBA(mol)
    except:
        return np.nan

def CalcNumLipinskiHBD(mol) -> int:
    try:
        return d.CalcNumLipinskiHBD(mol)
    except:
        return np.nan


def CalcNumRings(mol) -> int:
    try:
        return d.CalcNumRings(mol)
    except:
        return np.nan

def CalcNumRotatableBonds(mol) -> int:
    try:
        return d.CalcNumRotatableBonds(mol)
    except:
        return np.nan


def CalcNumSaturatedCarbocycles(mol) -> int:
    try:
        return d.CalcNumSaturatedCarbocycles(mol)
    except:
        return np.nan

def CalcNumSaturatedHeterocycles(mol) -> int:
    try:
        return d.CalcNumSaturatedHeterocycles(mol)
    except:
        return np.nan

def CalcNumSaturatedRings(mol) -> int:
    try:
        return d.CalcNumSaturatedRings(mol)
    except:
        return np.nan


def CalcNumSpiroAtoms(mol) -> int:
    try:
        return d.CalcNumSpiroAtoms(mol)
    except:
        return np.nan

def CalcNumUnspecifiedAtomStereoCenters(mol) -> int:
    try:
        return d.CalcNumUnspecifiedAtomStereoCenters(mol)
    except:
        return np.nan


def CalcPBF(mol) -> float:
    try:
        return d.CalcPBF(mol)
    except:
        return np.nan


def CalcPMI1(mol) -> float:
    try:
        return d.CalcPMI1(mol)
    except:
        return np.nan

def CalcPMI2(mol) -> float:
    try:
        return d.CalcPMI2(mol)
    except:
        return np.nan

def CalcPMI3(mol) -> float:
    try:
        return d.CalcPMI3(mol)
    except:
        return np.nan


def CalcPhi(mol) -> float:
    try:
        return d.CalcPhi(mol)
    except:
        return np.nan

def CalcRDF(mol) -> float:
    try:
        return d.CalcRDF(mol)
    except:
        return np.nan

def CalcRadiusOfGyration(mol) -> float:
    try:
        return d.CalcRadiusOfGyration(mol)
    except:
        return np.nan

def CalcSpherocityIndex(mol) -> float:
    try:
        return d.CalcSpherocityIndex(mol)
    except:
        return np.nan

def CalcTPSA(mol) -> float:
    try:
        return d.CalcTPSA(mol)
    except:
        return np.nan

def CalcWHIM(mol) -> float:
    try:
        return d.CalcWHIM(mol)
    except:
        return np.nan

def BCUT2D(mol):
    try:
        return d.BCUT2D(mol[0])
    except:
        return [np.nan]*8
    
def CalcAUTOCORR2D(mol):
    try:
        return d.CalcAUTOCORR2D(mol[0])
    except:
        return [np.nan]*192

def fr_Al_COO(mol):
    try:
        return f.fr_Al_COO(mol)
    except:
        return np.nan

def fr_Al_OH(mol):
    try:
        return f.fr_Al_OH(mol)
    except:
        return np.nan

def fr_Al_OH_noTert(mol):
    try:
        return f.fr_Al_OH_noTert(mol)
    except:
        return np.nan

def fr_ArN(mol):
    try:
        return f.fr_ArN(mol)
    except:
        return np.nan

def fr_Ar_COOR(mol):
    try:
        return f.fr_Ar_COOR(mol)
    except:
        return np.nan

def fr_Ar_N(mol):
    try:
        return f.fr_Ar_N(mol)
    except:
        return np.nan

def fr_Ar_NH(mol):
    try:
        return f.fr_Ar_NH(mol)
    except:
        return np.nan

def fr_Ar_OH(mol):
    try:
        return f.fr_Ar_OH(mol)
    except:
        return np.nan

def fr_COO(mol):
    try:
        return f.fr_COO(mol)
    except:
        return np.nan

def fr_COO2(mol):
    try:
        return f.fr_COO2(mol)
    except:
        return np.nan

def fr_C_O(mol):
    try:
        return f.fr_C_O(mol)
    except:
        return np.nan

def fr_C_O_noCOO(mol):
    try:
        return f.fr_C_O_noCOO(mol)
    except:
        return np.nan

def fr_C_S(mol):
    try:
        return f.fr_C_S(mol)
    except:
        return np.nan

def fr_HOCCN(mol):
    try:
        return f.fr_HOCCN(mol)
    except:
        return np.nan

def fr_Imine(mol):
    try:
        return f.fr_Imine(mol)
    except:
        return np.nan

def fr_NH0(mol):
    try:
        return f.fr_NH0(mol)
    except:
        return np.nan

def fr_NH1(mol):
    try:
        return f.fr_NH1(mol)
    except:
        return np.nan

def fr_NH2(mol):
    try:
        return f.fr_NH2(mol)
    except:
        return np.nan

def fr_N_O(mol):
    try:
        return f.fr_N_O(mol)
    except:
        return np.nan

def fr_Ndealkylation1(mol):
    try:
        return f.fr_Ndealkylation1(mol)
    except:
        return np.nan

def fr_Ndealkylation2(mol):
    try:
        return f.fr_Ndealkylation2(mol)
    except:
        return np.nan

def fr_Nhpyrrole(mol):
    try:
        return f.fr_Nhpyrrole(mol)
    except:
        return np.nan

def fr_SH(mol):
    try:
        return f.fr_SH(mol)
    except:
        return np.nan

def fr_aldehyde(mol):
    try:
        return f.fr_aldehyde(mol)
    except:
        return np.nan

def fr_alkyl_carbamate(mol):
    try:
        return f.fr_alkyl_carbamate(mol)
    except:
        return np.nan

def fr_alkyl_halide(mol):
    try:
        return f.fr_alkyl_halide(mol)
    except:
        return np.nan

def fr_allylic_oxid(mol):
    try:
        return f.fr_allylic_oxid(mol)
    except:
        return np.nan

def fr_amide(mol):
    try:
        return f.fr_amide(mol)
    except:
        return np.nan

def fr_amidine(mol):
    try:
        return f.fr_amidine(mol)
    except:
        return np.nan

def fr_aniline(mol):
    try:
        return f.fr_aniline(mol)
    except:
        return np.nan

def fr_aryl_methyl(mol):
    try:
        return f.fr_aryl_methyl(mol)
    except:
        return np.nan

def fr_azide(mol):
    try:
        return f.fr_azide(mol)
    except:
        return np.nan
    
def fr_azo(mol):
    try:
        return f.fr_azo(mol)
    except:
        return np.nan

def fr_barbitur(mol):
    try:
        return f.fr_barbitur(mol)
    except:
        return np.nan

def fr_benzene(mol):
    try:
        return f.fr_benzene(mol)
    except:
        return np.nan

def fr_benzodiazepine(mol):
    try:
        return f.fr_benzodiazepine(mol)
    except:
        return np.nan

def fr_bicyclic(mol):
    try:
        return f.fr_bicyclic(mol)
    except:
        return np.nan

def fr_diazo(mol):
    try:
        return f.fr_diazo(mol)
    except:
        return np.nan

def fr_dihydropyridine(mol):
    try:
        return f.fr_dihydropyridine(mol)
    except:
        return np.nan

def fr_epoxide(mol):
    try:
        return f.fr_epoxide(mol)
    except:
        return np.nan

def fr_ester(mol):
    try:
        return f.fr_ester(mol)
    except:
        return np.nan

def fr_ether(mol):
    try:
        return f.fr_ether(mol)
    except:
        return np.nan

def fr_furan(mol):
    try:
        return f.fr_furan(mol)
    except:
        return np.nan

def fr_guanido(mol):
    try:
        return f.fr_guanido(mol)
    except:
        return np.nan

def fr_halogen(mol):
    try:
        return f.fr_halogen(mol)
    except:
        return np.nan

def fr_hdrzine(mol):
    try:
        return f.fr_hdrzine(mol)
    except:
        return np.nan

def fr_hdrzone(mol):
    try:
        return f.fr_hdrzone(mol)
    except:
        return np.nan

def fr_imidazole(mol):
    try:
        return f.fr_imidazole(mol)
    except:
        return np.nan

def fr_imide(mol):
    try:
        return f.fr_imide(mol)
    except:
        return np.nan

def fr_isocyan(mol):
    try:
        return f.fr_isocyan(mol)
    except:
        return np.nan

def fr_isothiocyan(mol):
    try:
        return f.fr_isothiocyan(mol)
    except:
        return np.nan

def fr_ketone(mol):
    try:
        return f.fr_ketone(mol)
    except:
        return np.nan

def fr_ketone_Topliss(mol):
    try:
        return f.fr_ketone_Topliss(mol)
    except:
        return np.nan

def fr_lactam(mol):
    try:
        return f.fr_lactam(mol)
    except:
        return np.nan

def fr_lactone(mol):
    try:
        return f.fr_lactone(mol)
    except:
        return np.nan

def fr_methoxy(mol):
    try:
        return f.fr_methoxy(mol)
    except:
        return np.nan

def fr_morpholine(mol):
    try:
        return f.fr_morpholine(mol)
    except:
        return np.nan

def fr_nitrile(mol):
    try:
        return f.fr_nitrile(mol)
    except:
        return np.nan

def fr_nitro(mol):
    try:
        return f.fr_nitro(mol)
    except:
        return np.nan

def fr_nitro_arom(mol):
    try:
        return f.fr_nitro_arom(mol)
    except:
        return np.nan

def fr_nitro_arom_nonortho(mol):
    try:
        return f.fr_nitro_arom_nonortho(mol)
    except:
        return np.nan

def fr_nitroso(mol):
    try:
        return f.fr_nitroso(mol)
    except:
        return np.nan

def fr_oxazole(mol):
    try:
        return f.fr_oxazole(mol)
    except:
        return np.nan

def fr_oxime(mol):
    try:
        return f.fr_oxime(mol)
    except:
        return np.nan

def fr_para_hydroxylation(mol):
    try:
        return f.fr_para_hydroxylation(mol)
    except:
        return np.nan

def fr_phenol(mol):
    try:
        return f.fr_phenol(mol)
    except:
        return np.nan

def fr_phenol_noOrthoHbond(mol):
    try:
        return f.fr_phenol_noOrthoHbond(mol)
    except:
        return np.nan

def fr_phos_acid(mol):
    try:
        return f.fr_phos_acid(mol)
    except:
        return np.nan

def fr_phos_ester(mol):
    try:
        return f.fr_phos_ester(mol)
    except:
        return np.nan

def fr_piperdine(mol):
    try:
        return f.fr_piperdine(mol)
    except:
        return np.nan

def fr_piperzine(mol):
    try:
        return f.fr_piperzine(mol)
    except:
        return np.nan

def fr_priamide(mol):
    try:
        return f.fr_priamide(mol)
    except:
        return np.nan

def fr_prisulfonamd(mol):
    try:
        return f.fr_prisulfonamd(mol)
    except:
        return np.nan

def fr_pyridine(mol):
    try:
        return f.fr_pyridine(mol)
    except:
        return np.nan

def fr_quatN(mol):
    try:
        return f.fr_quatN(mol)
    except:
        return np.nan

def fr_sulfide(mol):
    try:
        return f.fr_sulfide(mol)
    except:
        return np.nan
    
def fr_sulfonamd(mol):
    try:
        return f.fr_sulfonamd(mol)
    except:
        return np.nan

def fr_sulfone(mol):
    try:
        return f.fr_sulfone(mol)
    except:
        return np.nan

def fr_term_acetylene(mol):
    try:
        return f.fr_term_acetylene(mol)
    except:
        return np.nan

def fr_tetrazole(mol):
    try:
        return f.fr_tetrazole(mol)
    except:
        return np.nan

def fr_thiazole(mol):
    try:
        return f.fr_thiazole(mol)
    except:
        return np.nan

def fr_thiocyan(mol):
    try:
        return f.fr_thiocyan(mol)
    except:
        return np.nan

def fr_thiophene(mol):
    try:
        return f.fr_thiophene(mol)
    except:
        return np.nan

def fr_unbrch_alkane(mol):
    try:
        return f.fr_unbrch_alkane(mol)
    except:
        return np.nan

def fr_urea(mol):
    try:
        return f.fr_urea(mol)
    except:
        return np.nan

def CalcFractionCSP3(mol):
    try:
        return l.CalcFractionCSP3(mol)
    except:
        return np.nan

def CalcNumAliphaticCarbocycles(mol):
    try:
        return l.CalcNumAliphaticCarbocycles(mol)
    except:
        return np.nan

def CalcNumAliphaticHeterocycles(mol):
    try:
        return l.CalcNumAliphaticHeterocycles(mol)
    except:
        return np.nan

def CalcNumAliphaticRings(mol):
    try:
        return l.CalcNumAliphaticRings(mol)
    except:
        return np.nan

def CalcNumAromaticCarbocycles(mol):
    try:
        return l.CalcNumAromaticCarbocycles(mol)
    except:
        return np.nan

def CalcNumAromaticHeterocycles(mol):
    try:
        return l.CalcNumAromaticHeterocycles(mol)
    except:
        return np.nan

def CalcNumAromaticRings(mol):
    try:
        return l.CalcNumAromaticRings(mol)
    except:
        return np.nan

def CalcNumSaturatedCarbocycles(mol):
    try:
        return l.CalcNumSaturatedCarbocycles(mol)
    except:
        return np.nan

def CalcNumSaturatedHeterocycles(mol):
    try:
        return l.CalcNumSaturatedHeterocycles(mol)
    except:
        return np.nan

def CalcNumSaturatedRings(mol):
    try:
        return l.CalcNumSaturatedRings(mol)
    except:
        return np.nan

def GetMorganFingerprint(mol):
    try:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol[0],2,nBits=124))
    except:
        return [np.nan]*124

def create_features():
    df_train = pd.read_csv('../data/training_smiles.csv', index_col=0)
    df_test = pd.read_csv('../data/test_smiles.csv', index_col=0)
    df_test['ACTIVE'] = np.nan
    df = pd.concat([df_train, df_test], axis=0)

    basic_features = [ 
    GetNumAtoms,
    ExactMolWt,
    CalcAsphericity,
    CalcChi0n,
    CalcChi0v,
    CalcChi1n,
    CalcChi1v,
    CalcChi2n,
    CalcChi2v,
    CalcChi3n,
    CalcChi3v,
    CalcChi4n,
    CalcChi4v,
    CalcEccentricity,
    CalcFractionCSP3,
    CalcHallKierAlpha,
    CalcInertialShapeFactor,
    CalcKappa1,
    CalcKappa2,
    CalcKappa3,
    CalcLabuteASA,
    CalcMORSE,
    CalcMolFormula,
    CalcNPR1,
    CalcNPR2,
    CalcNumAliphaticCarbocycles,
    CalcNumAliphaticHeterocycles,
    CalcNumAliphaticRings,
    CalcNumAmideBonds,
    CalcNumAromaticCarbocycles,
    CalcNumAromaticHeterocycles,
    CalcNumAromaticRings,
    CalcNumAtomStereoCenters,
    CalcNumAtoms,
    CalcNumBridgeheadAtoms,
    CalcNumHBA,
    CalcNumHBD,
    CalcNumHeavyAtoms,
    CalcNumHeteroatoms,
    CalcNumHeterocycles,
    CalcNumLipinskiHBA,
    CalcNumLipinskiHBD,
    CalcNumRings,
    CalcNumRotatableBonds,
    CalcNumSaturatedCarbocycles,
    CalcNumSaturatedHeterocycles,
    CalcNumSaturatedRings,
    CalcNumSpiroAtoms,
    CalcNumUnspecifiedAtomStereoCenters,
    CalcPBF,
    CalcPMI1,
    CalcPMI2,
    CalcPMI3,
    CalcPhi,
    CalcRDF,
    CalcRadiusOfGyration,
    CalcSpherocityIndex,
    CalcTPSA,
    CalcWHIM,
    ]

    framents_features = [
    fr_Al_COO,
    fr_Al_OH,
    fr_Al_OH_noTert,
    fr_ArN,
    fr_Ar_N,
    fr_Ar_NH,
    fr_Ar_OH,
    fr_COO,
    fr_COO2,
    fr_C_O,
    fr_C_O_noCOO,
    fr_C_S,
    fr_HOCCN,
    fr_Imine,
    fr_NH0,
    fr_NH1,
    fr_NH2,
    fr_N_O,
    fr_Ndealkylation1,
    fr_Ndealkylation2,
    fr_Nhpyrrole,
    fr_SH,
    fr_aldehyde,
    fr_alkyl_carbamate,
    fr_alkyl_halide,
    fr_allylic_oxid,
    fr_amide,
    fr_amidine,
    fr_aniline,
    fr_aryl_methyl,
    fr_azide,
    fr_azo,
    fr_barbitur,
    fr_benzene,
    fr_benzodiazepine,
    fr_bicyclic,
    fr_diazo,
    fr_dihydropyridine,
    fr_epoxide,
    fr_ester,
    fr_ether,
    fr_furan,
    fr_guanido,
    fr_halogen,
    fr_hdrzine,
    fr_hdrzone,
    fr_imidazole,
    fr_imide,
    fr_isocyan,
    fr_isothiocyan,
    fr_ketone,
    fr_ketone_Topliss,
    fr_lactam,
    fr_lactone,
    fr_methoxy,
    fr_morpholine,
    fr_nitrile,
    fr_nitro,
    fr_nitro_arom,
    fr_nitro_arom_nonortho,
    fr_nitroso,
    fr_oxazole,
    fr_oxime,
    fr_para_hydroxylation,
    fr_phenol,
    fr_phenol_noOrthoHbond,
    fr_phos_acid,
    fr_phos_ester,
    fr_piperdine,
    fr_piperzine,
    fr_priamide,
    fr_prisulfonamd,
    fr_pyridine,
    fr_quatN,
    fr_sulfide,
    fr_sulfonamd,
    fr_sulfone,
    fr_term_acetylene,
    fr_tetrazole,
    fr_thiazole,
    fr_thiocyan,
    fr_thiophene,
    fr_unbrch_alkane,
    fr_urea
    ]

    Lipinski_features = [
        CalcFractionCSP3,
        CalcNumAliphaticCarbocycles,
        CalcNumAliphaticHeterocycles,
        CalcNumAliphaticRings,
        CalcNumAromaticCarbocycles,
        CalcNumAromaticHeterocycles,
        CalcNumAromaticRings,
        CalcNumSaturatedCarbocycles,
        CalcNumSaturatedHeterocycles,
        CalcNumSaturatedRings
    ]

    df[MolFromSmiles.__name__] = df['SMILES'].apply(MolFromSmiles)

    for func in tqdm(basic_features):
        df[func.__name__] = df['MolFromSmiles'].apply(func)
    
    df_BCUT2D = df['MolFromSmiles'].to_frame().apply(BCUT2D, axis = 1, result_type = 'expand')
    df_BCUT2D.rename(lambda x: 'BCUT2D_' + str(x), axis = 1, inplace = True)
    assert(len(df) == len(df_BCUT2D))
    df = pd.concat([df, df_BCUT2D], axis = 1)

    df_AUTO2D = df['MolFromSmiles'].to_frame().apply(CalcAUTOCORR2D, axis = 1, result_type = 'expand')
    df_AUTO2D.rename(columns = lambda x : 'AUTO2D_' + str(x), inplace = True)
    assert(len(df) == len(df_AUTO2D))
    df = pd.concat([df, df_AUTO2D], axis = 1)

    for func in tqdm(framents_features):
        df[func.__name__] = df['MolFromSmiles'].apply(func)
    
    df_MorganFingerprint = df['MolFromSmiles'].to_frame().apply(GetMorganFingerprint, axis = 1, result_type = 'expand')
    df_MorganFingerprint.rename(columns = lambda x : 'MorganFingerprint_' + str(x), inplace = True)
    assert(len(df) == len(df_MorganFingerprint))
    df = pd.concat([df, df_MorganFingerprint], axis = 1)

    df.drop(['MolFromSmiles'], axis = 1, inplace = True)
    df.to_csv('../data/features.csv', index= False)

if __name__ == '__main__':
    create_features()