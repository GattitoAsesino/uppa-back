from ultralytics import YOLO
import numpy as np
import cv2


ESPECIES_CULTIVADAS  = {
    0: "Mycena metata", 1: "Mycena epipterygia", 2: "Clitocybe fragrans",
    3: "Mycena pseudocorticola", 4: "Mycena galericulata", 5: "Mycena tintinnabulum",
    6: "Mycena filopes", 7: "Mycena clavularis", 8: "Mycena arcangeliana",
    9: "Mycena chlorantha", 10: "Russula claroflava", 11: "Agaricus campestris",
    12: "Boletus reticulatus", 13: "Mycena sanguinolenta", 14: "Clitocybe costata",
    15: "Amanita rubescens", 16: "Agaricus xanthodermus", 17: "Russula vesca",
    18: "Amanita pantherina", 19: "Mycena renati", 20: "Boletus pinophilus",
    21: "Boletus edulis", 22: "Mycena citrinomarginata", 23: "Russula recondita",
    24: "Russula graveolens", 25: "Russula solaris", 26: "Russula violeipes",
    27: "Mycena acicula", 28: "Agaricus augustus", 29: "Amanita fulva",
    30: "Russula odorata", 31: "Russula olivacea", 32: "Russula nauseosa",
    33: "Amanita gemmata", 34: "Russula pelargonia", 35: "Russula versicolor",
    36: "Russula sororia", 37: "Agaricus moelleri", 38: "Russula amoenolens",
    39: "Agaricus sylvicola", 40: "Russula paludosa", 41: "Russula pseudointegra",
    42: "Agaricus arvensis", 43: "Agaricus impudicus", 44: "Russula grata",
    45: "Mycena purpureofusca", 46: "Amanita phalloides", 47: "Russula virescens",
    48: "Russula romellii", 49: "Russula velenovskyi", 50: "Russula ochroleuca",
    51: "Russula carpini", 52: "Russula melzeri", 53: "Russula delica",
    54: "Amanita virosa", 55: "Mycena stipata", 56: "Agaricus bitorquis",
    57: "Russula acrifolia", 58: "Russula rosea", 59: "Agaricus subperonatus",
    60: "Amanita muscaria", 61: "Russula cyanoxantha", 62: "Amanita franchetii",
    63: "Russula subfoetens", 64: "Mycena haematopus", 65: "Amanita crocea",
    66: "Amanita citrina", 67: "Mycena rubromarginata", 68: "Agaricus litoralis",
    69: "Russula betularum", 70: "Agaricus bernardii", 71: "Russula chloroides",
    72: "Mycena galopus", 73: "Mycena rosea", 74: "Russula queletii",
    75: "Russula adusta", 76: "Russula depallens", 77: "Agaricus sylvaticus",
    78: "Agaricus crocodilinus", 79: "Russula luteotacta", 80: "Russula sanguinea",
    81: "Russula gracillima", 82: "Russula decolorans", 83: "Russula silvestris",
    84: "Russula xerampelina", 85: "Russula rhodopus", 86: "Russula aurea",
    87: "Mycena crocata", 88: "Russula risigallina", 89: "Russula sardonia",
    90: "Russula foetens", 91: "Russula atropurpurea", 92: "Russula curtipes",
    93: "Russula nobilis", 94: "Mycena pelianthina", 95: "Mycena luteovariegata",
    96: "Amanita porphyria", 97: "Mycena flavescens", 98: "Clitocybe phyllophila",
    99: "Russula fellea", 100: "Russula nitida", 101: "Russula anthracina",
    102: "Russula roseoaurantia", 103: "Clitocybe odora", 104: "Russula puellula",
    105: "Mycena tenerrima", 106: "Agaricus bohusii", 107: "Mycena inclinata",
    108: "Agaricus depauperatus", 109: "Russula vinosa", 110: "Mycena rosella",
    111: "Russula atrorubens", 112: "Mycena pterigena", 113: "Mycena aetites",
    114: "Mycena zephirus", 115: "Clitocybe rivulosa", 116: "Russula laccata",
    117: "Mycena maculata", 118: "Mycena erubescens", 119: "Mycena polygramma",
    120: "Clitocybe nebularis", 121: "Mycena leptocephala", 122: "Mycena amicta",
    123: "Russula fragilis", 124: "Mycena juniperina", 125: "Mycena vitilis",
    126: "Mycena meliigena", 127: "Mycena cinerella", 128: "Clitocybe metachroa",
    129: "Mycena xantholeuca", 130: "Russula puellaris", 131: "Mycena bulbosa",
    132: "Amanita olivaceogrisea", 133: "Amanita ceciliae", 134: "Russula aeruginea",
    135: "Russula brunneoviolacea", 136: "Russula integra", 137: "Clitocybe agrestis",
    138: "Amanita regalis", 139: "Clitocybe phaeophthalma", 140: "Russula illota",
    141: "Russula emetica", 142: "Russula alnetorum", 143: "Russula cessans",
    144: "Russula insignis", 145: "Russula maculata", 146: "Mycena stylobates",
    147: "Mycena capillaripes", 148: "Russula parazurea", 149: "Mycena olivaceomarginata",
    150: "Russula mustelina", 151: "Russula heterophylla", 152: "Russula viscida",
    153: "Mycena aurantiomarginata", 154: "Agaricus brunneolus", 155: "Russula aurora",
    156: "Russula laeta", 157: "Agaricus langei", 158: "Russula grisea",
    159: "Russula densifolia", 160: "Russula melliolens", 161: "Amanita lividopallescens",
    162: "Russula faustiana", 163: "Russula ionochlora", 164: "Russula caerulea",
    165: "Amanita submembranacea", 166: "Clitocybe vibecina", 167: "Clitocybe amarescens",
    168: "Mycena pseudopicta", 169: "Mycena smithiana", 170: "Mycena abramsii",
    171: "Russula cuprea", 172: "Russula farinipes", 173: "Russula veternosa",
    174: "Russula seperina", 175: "Russula aquosa", 176: "Agaricus bisporus",
    177: "Russula faginea", 178: "Russula clavipes", 179: "Russula subrubens",
    180: "Mycena clavicularis", 181: "Agaricus cupreobrunneus", 182: "Mycena polyadelpha",
    183: "Mycena silvae-nigrae", 184: "Amanita vaginata", 185: "Mycena picta",
    186: "Amanita huijsmanii", 187: "Mycena vulgaris", 188: "Clitocybe subspadicea",
    189: "Mycena belliae", 190: "Agaricus dulcidulus", 191: "Russula turci",
    192: "Clitocybe diatreta", 193: "Clitocybe obsoleta", 194: "Mycena mucor",
    195: "Amanita simulans", 196: "Mycena megaspora", 197: "Mycena pearsoniana",
    198: "Agaricus devoniensis", 199: "Mycena albidolilacea", 200: "Agaricus subfloccosus",
    201: "Mycena latifolia", 202: "Agaricus altipes", 203: "Agaricus lanipes",
    204: "Russula emeticicolor", 205: "Russula albonigra", 206: "Mycena scirpicola",
    207: "Clitocybe barbularum", 208: "Mycena concolor", 209: "Mycena mirata"
}

model = YOLO("my_model.pt")

def detectar_especie(image_bytes: bytes) -> dict:
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    results = model.predict(image, verbose=False)

    detecciones = []
    for result in results:
        for box in result.boxes:
            clase_id  = int(box.cls[0])
            confianza = float(box.conf[0])
            nombre    = model.names[clase_id]
            detecciones.append({
                "clase_id":  clase_id,
                "especie":   nombre,
                "confianza": round(confianza * 100, 1),
                "cultivada": clase_id in ESPECIES_CULTIVADAS
            })

    if not detecciones:
        return {
            "detectado": False,
            "mensaje": "No se detectó ningún hongo en la imagen"
        }

    mejor = max(detecciones, key=lambda x: x["confianza"])
    return {
        "detectado":  True,
        'id_especie':   mejor["clase_id"],  
        "especie":    mejor["especie"],
        "confianza":  mejor["confianza"],
        "cultivada":  mejor["cultivada"],
        "mensaje":    f"{'Especie cultivable' if mejor['cultivada'] else 'Hongo silvestre'}: {mejor['especie']}"
    }