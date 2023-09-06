#[[[cog
#import cog
#import os
#chart_item_dict = {
#    'AF0': 224690,
#    'AF1': 220210,
#    'BE_arteriell': 224828,
#    'Compliance': 229661,
#    'DAP': 220051,
#    'FiO2': 223835,
#    'GOT' : 220587,
#    'GPT' : 220644,
#    'HF' :  220045,
#    'Expiratory Ratio' : 226871,
#    'Inspiratory Ratio' : 226873,
#    'Koerperkerntemperatur F' : 223761,
#    'Koerperkerntemperatur C' : 223762, 
#    'MAP': 220052,
#    'PEEP' : 220339,
#    'SAP' : 220050,
#    'SaO2': 220227,
#    'SpO2': 220277,
#    'SzvO2': 223772,
#    'ZVD': 220074,
#    'pH_arteriell': 223830,
#    'paCO2_(ohne_Temp-Korrektur)': 220235,
#    'paO2_(ohne_Temp-Korrektur)' : 220224,
#    'EVLWI': 228179,
#    'CK-MB_MIMIC' : 227445,
#    'PCWP' : 223771,
#    'MPAP' : 220061,
#    'SV_kontinuierlich0': 227547,
#    'SV_kontinuierlich1': 228374,
#    'HZV_(kontinuierlich)': 224842,
#    'BNP_MIMIC': 227446,
#    'Inhalatives_NO' : 224749,
#    'DPAP': 220060,
#    'SVRI' : 228185,
#    'GEDVI' : 228180,
#    'P_EI' : 224696,
#    'SPAP' : 220059,
#    'etCO2': 228640,
#    'Tidalvolumen' : 224685,
#    'Tidalvolumen_spont0': 224686,
#    'Tidalvolumen_spont1': 224421,
#    'SVI_(kontinuierlich)' : 228182,
#    'AF_spontan0' : 224689,
#    'AF_spontan1' : 224422,
#    'LDH_MIMIC' : 220632,
#    'Lymphozyten_absolut': 229358,
#    'HI_(kontinuierlich)' : 228368,
#    'Lagerungstherapie' : 224093,
#    'ECMO' : 229267,
#    'Extrakorporaler_Gasfluss_(O2)': 229278,
#    'Gaszusammensetzung_(%O2)': 229280,
#    'Extrakorporaler_Blutfluss': 229270,
#    'D-Dimere' : 225636
#}
#
#lab_item_dict = {
#    'Amylase': 50867,
#    'Bilirubin_ges': 50885,
#    'CK': 50910,
#    'Harnstoff' : 51006,
#    'Haematokrit' : 51221,
#    'Haemoglobin0' : 51222,
#    'Haemoglobin1'  : 50811,
#    'INR'   :   51237,
#    'Kreatinin': 50912,
#    'Leukozyten' : 51301,
#    'Lipase_MIMIC': 50956,
#    'pTT': 51275,
#    'Troponin' : 51003,
#    'Thrombozyten' : 51265,
#    'Albumin' : 50862,
#    'CRP' : 50889,
#    'NT-pro_BNP': 50963
#}
#
#input_item_dict = {
#    'Furosemid_intravenoes_kontinuierlich' : 221794,
#    'Norepinephrin_intravenoes_kontinuierlich' : 221906,
#    'Propofol_intravenoes_kontinuierlich' : 222168,
#    'Vasopressin_intravenoes_kontinuierlich' : 222315,
#    'Rocuronium_intravenoes_bolusweise' : 229233,
#    'Midazolam_intravenoes_kontinuierlich' : 221668,
#    'Morphin_intravenoes_kontinuierlich' : 225154,
#    'Milrinon_intravenoes_kontinuierlich' : 221986,
#    'Fentanyl_intravenoes_kontinuierlich' : 221744,
#    'Dobutamin_intravenoes_kontinuierlich' : 221653,
#    'Ketanest_intravenoes_kontinuierlich' : 221712,
#    'Epinephrin_intravenoes_kontinuierlich' : 221289,
#    'Dexmedetomidin_intravenoes_kontinuierlich' : 229420,
#}
#procedure_item_dict = {
#    'Sevofluran_inhalativ' : 229984,
#    'Isofluran_inhalativ' : 229983
#}
#
#
#
#mimiciv_mapping_rev = {**chart_item_dict, **lab_item_dict, **input_item_dict, **procedure_item_dict}
#
#
#
#lab_keys = list(lab_item_dict)
#lab_values = list(lab_item_dict.values())
#chart_keys = list(chart_item_dict)
#chart_values = list(chart_item_dict.values())
#input_keys = list(input_item_dict)
#input_values = list(input_item_dict.values())
#procedure_keys = list(procedure_item_dict)
#procedure_values = list(procedure_item_dict.values())
#mapping_keys = list(mimiciv_mapping_rev)
#mapping_values = list(mimiciv_mapping_rev.values())
#cog.outl("chart_item_dict = {")
#for i in range(len(chart_keys)):
#    if i + 1 == len(chart_keys):
#        cog.outl("'%s': %s" % (chart_keys[i], chart_values[i]))
#    else :
#        cog.outl("'%s': %s," % (chart_keys[i], chart_values[i]))
#cog.outl("}")
#cog.outl("lab_item_dict = {")
#for i in range(len(lab_keys)):
#    if i + 1 == len(lab_keys):
#        cog.outl("'%s': %s" % (lab_keys[i], lab_values[i]))
#    else :
#        cog.outl("'%s': %s," % (lab_keys[i], lab_values[i]))
#cog.outl("}")
#cog.outl("input_item_dict = {")
#for i in range(len(input_keys)):
#    if i + 1 == len(input_keys):
#        cog.outl("'%s': %s" % (input_keys[i], input_values[i]))
#    else :
#        cog.outl("'%s': %s," % (input_keys[i], input_values[i]))
#cog.outl("}")
#cog.outl("procedure_item_dict = {")
#for i in range(len(procedure_keys)):
#    if i + 1 == len(procedure_keys):
#        cog.outl("'%s': %s" % (procedure_keys[i], procedure_values[i]))
#    else :
#        cog.outl("'%s': %s," % (procedure_keys[i], procedure_values[i]))
#cog.outl("}")
#cog.outl("lab_item_dict_rev = {")
#for i in range(len(lab_keys)):
#    if i == len(lab_keys) -1:
#        cog.outl("%s : '%s'" % (lab_values[i], lab_keys[i]))
#    else:
#        cog.outl("%s : '%s'," % (lab_values[i], lab_keys[i]))
#cog.outl("}")
#
#
#cog.outl("lab_index_dict = {")
#for i in range(len(lab_keys)):
#    if i == len(lab_keys) -1:
#        cog.outl("%s : %s" % (lab_values[i], i))
#    else:
#        cog.outl("%s : %s," % (lab_values[i], i))
#cog.outl("}")
#
# 
#cog.outl("lab_index_dict_rev = {")
#for i in range(len(lab_keys)):
#    if i == len(lab_keys) -1:
#        cog.outl("%s : %s" % (i, lab_values[i]))
#    else:
#        cog.outl("%s : %s," % (i, lab_values[i]))
#cog.outl("}")
#
#cog.outl("chart_item_dict_rev = {")
#for i in range(len(chart_keys)):
#    if i == len(chart_keys) -1:
#        cog.outl("%s : '%s'" % (chart_values[i], chart_keys[i]))
#    else:
#        cog.outl("%s : '%s'," % (chart_values[i], chart_keys[i]))
#cog.outl("}")
#
#cog.outl("chart_index_dict = {")
#for i in range(len(chart_keys)):
#    if i == len(chart_keys) -1:
#        cog.outl("%s : %s" % (chart_values[i], i))
#    else:
#        cog.outl("%s : %s," % (chart_values[i], i))
#cog.outl("}")
#
# 
#cog.outl("chart_index_dict_rev = {")
#for i in range(len(chart_keys)):
#    if i == len(chart_keys) -1:
#        cog.outl("%s : %s" % (i, chart_values[i]))
#    else:
#        cog.outl("%s : %s," % (i, chart_values[i]))
#cog.outl("}")
#
#cog.outl("input_item_dict_rev = {")
#for i in range(len(input_keys)):
#    if i == len(input_keys) -1:
#        cog.outl("%s : '%s'" % (input_values[i], input_keys[i]))
#    else:
#        cog.outl("%s : '%s'," % (input_values[i], input_keys[i]))
#cog.outl("}")
#
#cog.outl("input_index_dict = {")
#for i in range(len(input_keys)):
#    if i == len(input_keys) -1:
#        cog.outl("%s : %s" % (input_values[i], i))
#    else:
#        cog.outl("%s : %s," % (input_values[i], i))
#cog.outl("}")
#
#
#cog.outl("input_index_dict_rev = {")
#for i in range(len(input_keys)):
#    if i == len(input_keys) -1:
#        cog.outl("%s : %s" % (i, input_values[i]))
#    else:
#        cog.outl("%s : %s," % (i, input_values[i]))
#cog.outl("}")
#
#cog.outl("procedure_item_dict_rev = {")
#for i in range(len(procedure_keys)):
#    if i == len(procedure_keys) -1:
#        cog.outl("%s : '%s'" % (procedure_values[i], procedure_keys[i]))
#    else:
#        cog.outl("%s : '%s'," % (procedure_values[i], procedure_keys[i]))
#cog.outl("}")
#
#cog.outl("procedure_index_dict = {")
#for i in range(len(procedure_keys)):
#    if i == len(procedure_keys) -1:
#        cog.outl("%s : %s" % (procedure_values[i], i))
#    else:
#        cog.outl("%s : %s," % (procedure_values[i], i))
#cog.outl("}")
#
#
#cog.outl("procedure_index_dict_rev = {")
#for i in range(len(procedure_keys)):
#    if i == len(procedure_keys) -1:
#        cog.outl("%s : %s" % (i, procedure_values[i]))
#    else:
#        cog.outl("%s : %s," % (i, procedure_values[i]))
#cog.outl("}")
#cog.outl("#FIXME adjust AF, Temperatur, Tidalvolumen_spont, AF_spont SV_(kontinuierlich), Haemoglobin")
#cog.outl("mimiciv_mapping = {")
#for i in range(len(mapping_keys)):
#    if i == len(mapping_keys) -1:
#        cog.outl("%s : '%s'" % (mapping_values[i], mapping_keys[i]))
#    else:
#        cog.outl("%s : '%s'," % (mapping_values[i], mapping_keys[i]))
#cog.outl("}")
#]]]
#[[[end]]]
