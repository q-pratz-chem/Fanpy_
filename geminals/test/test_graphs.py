from geminals.graphs import generate_complete_pmatch, generate_biclique_pmatch

def test_generate_complete_pmatch():
    occ_indices = [0,1,3,4]
    answer = [ ((0,1), (3,4)),
               ((0,3), (1,4)),
               ((0,4), (1,3))]
    assert answer == list(generate_complete_pmatch(occ_indices))
    occ_indices = [0,1,3,4,6,7]
    answer = [ ((0,1), (3,4), (6,7)),
               ((0,1), (3,6), (4,7)),
               ((0,1), (3,7), (4,6)),
               ((0,3), (1,4), (6,7)),
               ((0,3), (1,6), (4,7)),
               ((0,3), (1,7), (4,6)),
               ((0,4), (1,3), (6,7)),
               ((0,4), (1,6), (3,7)),
               ((0,4), (1,7), (3,6)),
               ((0,6), (1,3), (4,7)),
               ((0,7), (1,3), (4,6)),
               ((0,6), (1,4), (3,7)),
               ((0,7), (1,4), (3,6)),
               ((0,6), (1,7), (3,4)),
               ((0,7), (1,6), (3,4)),
    ]
    assert answer == [tuple(sorted(i, key=lambda x: x[0])) for i in sorted(generate_complete_pmatch(occ_indices), key=lambda x:x[0])]


def test_generate_biclique_pmatch():
    indices_one = [0, 1]
    indices_two = [3, 4]
    answer = [ ((0,3), (1,4)),
               ((0,4), (1,3))]
    assert answer == list(generate_biclique_pmatch(indices_one, indices_two))
    indices_one = [0, 1, 3]
    indices_two = [4, 6, 7]
    answer = [ ((0,4), (1,6), (3,7)),
               ((0,4), (1,7), (3,6)),
               ((0,6), (1,4), (3,7)),
               ((0,6), (1,7), (3,4)),
               ((0, 7), (1, 4), (3, 6)),
               ((0,7), (1,6), (3,4)),
    ]
    assert answer == list(generate_biclique_pmatch(indices_one, indices_two))
