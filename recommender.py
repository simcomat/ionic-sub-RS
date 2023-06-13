from typing import Union
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from gensim.models import KeyedVectors
import json


def get_as_label(structure: Structure, symprec=0.1) -> str:
    """
    Generate a label for a given crystal structure based on its symmetry properties.

    The label is generated as a string in the following format: 
    "spacegroup_number_number_of_sites_wyckoff_letters(symmetry_info)".
    Wyckoff letters are used to describe the symmetry of atomic positions within the crystal structure.
    Each unique Wyckoff position corresponds to a different site symmetry.

    The function uses pymatgen's SpacegroupAnalyzer to perform space group analysis and to get the symmetrized 
    structure, wyckoff symbols, and symmetry dataset. 

    Parameters
    ----------
    structure : Structure
        The structure for which the label will be generated. Structure is a class from pymatgen's 
        core structure module, used to represent a crystal structure.
    symprec : float, optional
        The tolerance for symmetry finding. Default is 0.1.

    Returns
    -------
    str
        A label for the crystal structure. If an error is encountered during label generation,
        the function returns the string 'error'.
    """
    try:
        nsites = structure.num_sites
        sg_a = SpacegroupAnalyzer(structure, symprec=symprec)
        sym_struct = sg_a.get_symmetrized_structure()
        wyckoff_symbols = sym_struct.wyckoff_symbols
        wyckoff_symbols.sort(key=lambda x: x[1])
        sym_data = sg_a.get_symmetry_dataset()
        sg_number = sym_data['number']
        wyckoffs = sym_data['wyckoffs']
        site_syms = sym_data['site_symmetry_symbols']

        # mapping of wyckoff letters to their symmetries
        wyckoff_to_site_sym = dict(set(zip(wyckoffs, site_syms)))
        new_wyckoff_symbols = []
        for w in wyckoff_symbols:
            # replacing wyckoff symbols to their respective site symmetries
            new = ''.join(f'{w[:-1]}({wyckoff_to_site_sym.get(w[-1])})')
            new_wyckoff_symbols.append(new)

        # old wyckoff_symbols: ['1a', '1b', '3c']
        # new_wyckoff_symbols: ['1(m-3m)', '1(m-3m)', '3(m-3m)']
        # print(wyckoff_to_site_sym, wyckoff_symbols, new_wyckoff_symbols)
        label = f"{sg_number}_{nsites}_{'-'.join(new_wyckoff_symbols)}"

        return label
    except:
        return 'error'


class AMSite:
    """
    A class used to represent an Anonymous Motif (AM) site.

    Attributes
    ----------
    data : dict
        A dictionary containing all information about the site.
    label : str
        The label for the site.
    AM_label : str
        The Anonymous Motif label for the site.
    space_group : int
        The space group number.
    nsites : int
        The number of sites.
    site_sym : tuple
        A tuple of site symmetry information.
    example_structure_index : int
        The index of the example structure.
    """

    def __init__(self, AMSite_dict: dict):
        self.data = AMSite_dict
        self.label = AMSite_dict['label']
        self.AM_label = self.label.split('[')[0]
        self.space_group = int(self.label.split('_')[0])
        self.nsites = int(self.label.split('_')[1])
        site_sym_str = self.label.split(':')[1][:-1]
        site_sym_parts = site_sym_str.split('(')
        self.site_sym = (int(site_sym_parts[0]), site_sym_parts[1][:-1])
        self.example_structure_index = AMSite_dict['site_index']


class AnonymousMotif:
    """
    A class used to represent an Anonymous Motif (AM).

    Attributes
    ----------
    label : str
        The label for the AM.
    space_group : int
        The space group number.
    nsites : int
        The number of sites.
    wyckoff_list : list
        A list of Wyckoff positions.
    subgroup_number : int
        The subgroup number.
    example_structure : dict
        The example structure.
    equivalent_sites_indexes : list
        The equivalent sites indexes.
    sites : list
        A list of AMSite objects.
    data : dict
        A dictionary containing all information about the AM except the sites.
    """

    def __init__(self, AM_dict: dict):
        self.label = AM_dict['AM_label']
        self.space_group = int(self.label.split('_')[0])
        self.nsites = int(self.label.split('_')[1])
        self.wyckoff_list = [(int(s.split('(')[0]), s.split(
            '(')[1][:-1]) for s in self.label.split('_')[2].split('-')]
        self.subgroup_number = int(self.label.split('_')[-1])
        self.example_structure = AM_dict['AM_example_structure']
        self.equivalent_sites_indexes = self.example_structure.site_properties['as_eq_sites']
        self.sites = [AMSite(site_dict) for site_dict in AM_dict['sites']]
        data = AM_dict.copy()
        del data['sites']
        self.data = data


class OccupationData:
    """
    A class used to handle occupation data.

    Attributes
    ----------
    data : dict
        The data from a json file.
    AM_list : list
        A list of Anonymous Motif dictionaries.
    AM_sites_list : list
        A list of labels of the sites.
    """

    def __init__(self, json_file_path: str):
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        self.AM_list = self.data['AM_list']
        self.AM_sites_list = [site['label'] for site in self.AM_list]

    def get_AnonymousMotif(self, AM_label):
        """
        Returns the AnonymousMotif object with the given label.

        Parameters
        ----------
        AM_label : str
            The label of the AM to be returned.

        Returns
        -------
        AnonymousMotif
            The AnonymousMotif object with the given label, or None if no such AM exists.
        """
        for AM in self.AM_list:
            if AM_label in AM['AM_label']:
                return AnonymousMotif(AM)
        return None

    def get_AM_from_OQMD_id(self, id) -> Union[AnonymousMotif, None]:
        """
        Returns the AnonymousMotif object with the given id.

        Parameters
        ----------
        id : str
            The id of the AM to be returned.

        Returns
        -------
        AnonymousMotif
            The AnonymousMotif object with the given id, or None if no such AM exists.
        """
        for AM in self.AM_list:
            if id in AM['entries_ids']:
                return AnonymousMotif(AM)
        return None

    def get_AM_from_structure(self, structure: Structure) -> Union[AnonymousMotif, None]:
        """
        Returns the AnonymousMotif object with the given structure.

        Parameters
        ----------
        structure : Structure
            The structure of the AM to be returned.

        Returns
        -------
        AnonymousMotif
            The AnonymousMotif object with the given structure, or None if no such AM exists.
        """
        struct_AM_label = get_as_label(structure)
        for AM in self.AM_list:
            vanilla_label = '_'.join(AM['AM_label'].split('-')[:-1])
            if vanilla_label == struct_AM_label:
                example_structure = AM['AM_example_structure']
                sm = StructureMatcher(stol=0.15)
                if sm.fit_anonymous(example_structure, structure):
                    return AnonymousMotif(AM)
        return None

    def get_AMSite(self, AM_site_label):
        """
        Returns the AMSite object with the given label.

        Parameters
        ----------
        AM_site_label : str
            The label of the AMSite to be returned.

        Returns
        -------
        AMSite
            The AMSite object with the given label, or None if no such site exists.
        """
        AM_label = AM_site_label.split['['][0]
        for AM in self.AM_list:
            if AM['AM_label'] == AM_label:
                for site in AM['sites']:
                    if site['label'] == AM_site_label:
                        return AMSite(site)
        return None


class RecommenderSystem:
    """
    RecommenderSystem class provides recommendations based on distance in embedding space.

    Attributes:
    occupation_data (OccupationData): An instance of OccupationData.
    distance_threshold (float): The distance threshold for recommendations.
    embedding (KeyedVectors): Word embeddings.
    ions_embedding (KeyedVectors): Word embeddings of ions.
    sites_embedding (KeyedVectors): Word embeddings of sites.
    """

    def __init__(self, occupation_data: OccupationData,
                 embedding: KeyedVectors,
                 distance_threshold: float):
        self.occupation_data = occupation_data
        self.distance_threshold = distance_threshold
        self.embedding = embedding

        ions, ions_wv = [], []
        sites, sites_wv = [], []
        for node_label in embedding.index_to_key:
            if len(node_label) > 2:
                sites.append(node_label)
                sites_wv.append(embedding[node_label])
            else:
                ions.append(node_label)
                ions_wv.append(embedding[node_label])

        ions_embedding = KeyedVectors(vector_size=embedding.vector_size)
        ions_embedding.add_vectors(ions, ions_wv)
        self.ions_embedding = ions_embedding

        sites_embedding = KeyedVectors(vector_size=embedding.vector_size)
        sites_embedding.add_vectors(sites, sites_wv)
        self.sites_embedding = sites_embedding

    def get_recommendation_for_ion(self, ion: str) -> list[tuple[str, float]]:
        """
        Returns a list of recommended sites for the given ion based on distance in the embedding space.

        Args:
        ion (str): The ion for which the recommendation is to be made.

        Returns:
        list[tuple[str, float]]: A list of tuples. Each tuple contains a recommended site and the corresponding distance.
        """
        assert ion in self.ions_embedding.index_to_key

        recommendation_list = []
        for site in self.sites_embedding.index_to_key:
            dist = self.embedding.distance(ion, site)
            if dist < self.distance_threshold:
                recommendation_list.append((site, dist))

        return recommendation_list

    def get_recommendation_for_site(self, site: str) -> list[tuple[str, float]]:
        """
        Returns a list of recommended ions for the given site based on distance in the embedding space.

        Args:
        site (str): The site for which the recommendation is to be made.

        Returns:
        list[tuple[str, float]]: A list of tuples. Each tuple contains a recommended ion and the corresponding distance.
        """
        assert site in self.sites_embedding.index_to_key

        recommendation_list = []
        for ion in self.ions_embedding.index_to_key:
            dist = self.embedding.distance(site, ion)
            if dist < self.distance_threshold:
                recommendation_list.append((ion, dist))

        return recommendation_list

    def get_recommendation_for_AM(self, AM: AnonymousMotif) -> dict[AMSite, list[tuple[str, float]]]:
        """
        Returns a dictionary of site-specific recommendations for an anonymous motif.

        Args:
        AM (AnonymousMotif): The anonymous motif for which recommendations are to be made.

        Returns:
        dict[AMSite, list[tuple[str, float]]]: A dictionary where keys are sites and values are lists of tuples. Each tuple contains a recommended ion and the corresponding distance.
        """
        recommendations = dict()
        for site in AM.sites:
            recommendations[site] = self.get_recommendation_for_site(site)
        return recommendations
