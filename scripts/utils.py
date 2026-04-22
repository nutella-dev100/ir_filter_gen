def tokens_to_structure(tokens):
    materials = []
    thicknesses = []

    for t in tokens:
        if t in ['BOS', 'EOS']:
            continue
        mat, thick = t.split('_')
        materials.append(mat)
        thicknesses.append(float(thick))

    return materials, thicknesses