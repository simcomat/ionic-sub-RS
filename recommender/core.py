from __future__ import annotations
from typing import Union
from pymatgen.core import Structure, Element
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from gensim.models import KeyedVectors
import json
import gzip


def get_anonymous_struct(s):
    s1 = s.copy()
    s1.replace_species({e: Element('C') for e in set(s.species)})
    return s1.get_primitive_structure()


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
        label = f"{sg_number}_{nsites}_{'_'.join(new_wyckoff_symbols)}"

        return label
    except:
        return None


class AMSite:
    """
    A class used to represent an Anonymous Motif (AM) site.

    Attributes
    ----------
    data : dict
        A dictionary containing all the details about the site.
    label : str
        A unique label identifying the site.
    AM_label : str
        The label of the Anonymous Motif (AM) to which the site belongs.
    space_group : int
        The space group number associated with the site.
    nsites : int
        The number of sites present in the site's AM.
    site_sym : tuple
        A tuple containing site symmetry information.
    site_index : int
        The index of the site in the example structure.
    """

    def __init__(self, AMSite_dict: dict):
        """
        Parameters
        ----------
        AMSite_dict : dict
            A dictionary containing details about the site.
        """
        self.data = AMSite_dict
        self.label = AMSite_dict['label']
        self.AM_label = self.label.split('[')[0]
        self.space_group = int(self.label.split('_')[0])
        self.nsites = int(self.label.split('_')[1])
        site_sym_str = self.label.split(':')[1][:-1]
        site_sym_parts = site_sym_str.split('(')
        self.site_sym = (int(site_sym_parts[0]), site_sym_parts[1][:-1])
        self.subgroup_number = int(self.AM_label.split('_')[-1])
        self.site_index = AMSite_dict['site_index']
        self.CNNf_median = AMSite_dict['CNNf_median']

    def __repr__(self):
        """
        Returns a string representation of the AMSite object.

        The representation includes the space group, number of sites, site symmetry, subgroup number and the site index.

        Returns
        -------
        str
            A string representation of the AMSite object.
        """
        return f'{(self.space_group, self.nsites, self.site_sym, self.subgroup_number)}\nsite index: {self.site_index}'


class AnonymousMotif:
    """
    A class used to represent an Anonymous Motif (AM) that abstracts the crystallographic structure of a motif.

    This class represents an AM, which is a prototype crystal structure motif, by encapsulating details such as 
    space group, number of sites, Wyckoff positions, and the example structure. The data are stored both in 
    attribute form and as a dictionary.

    Attributes
    ----------
    label : str
        A unique label that identifies the AM.
    space_group : int
        The space group number associated with the AM.
    nsites : int
        The number of sites present in the AM.
    wyckoff_list : list
        A list of Wyckoff positions included in the AM.
    subgroup_number : int
        The subgroup number associated with the AM.
    example_structure : dict
        An example structure of the AM represented as a dictionary.
    equivalent_sites_indexes : list
        A list of equivalent site indexes present in the AM.
    sites : list[AMSite]
        A list of AMSite objects each representing a site in the AM.
    data : dict
        A dictionary containing all attributes of the AM.
    """

    def __init__(self, AM_dict: dict):
        """
        Parameters
        ----------
        AM_dict : dict
            A dictionary containing details about the AM.
        """
        self.label = AM_dict['AM_label']
        self.space_group = int(self.label.split('_')[0])
        self.nsites = int(self.label.split('_')[1])

        wyckoff_list_parts = self.label.split('_')
        wyckoff_list_segment = '_'.join(wyckoff_list_parts[2:-1])
        self.wyckoff_list = [(int(s.split('(')[0]), s.split(
            '(')[1][:-1]) for s in wyckoff_list_segment.split('_')]

        self.subgroup_number = int(self.label.split('_')[-1])

        example_structure_dict = AM_dict['AM_example_structure']
        self.example_structure = Structure.from_str(
            example_structure_dict['structure'], fmt='json')
        self.equivalent_sites_indexes = example_structure_dict['eq_sites']

        self.sites = [AMSite(site_dict) for site_dict in AM_dict['sites']]
        self.entries_ids = AM_dict['entries_ids']
        data = AM_dict.copy()
        del data['sites']
        self.data = data

    def __repr__(self):
        """
        Returns a string representation of the AnonymousMotif object.

        The representation includes the space group, number of sites, Wyckoff positions, and the subgroup number.

        Returns
        -------
        str
            A string representation of the AnonymousMotif object.
        """
        return f'{(self.space_group, self.nsites, self.wyckoff_list, self.subgroup_number)}'


class OccupationData:
    """
    A class used to handle occupation data.

    Attributes
    ----------
    data : dict
        The data from the json file.
    AM_list : list
        A list of Anonymous Motif dictionaries.
    """

    def __init__(self, json_file_path: str):
        with gzip.open(json_file_path, 'rb') as file:
            data = file.read()
        self.data = json.loads(data.decode('utf-8'))
        # with open(json_file_path, 'r') as f:
        #     self.data = json.load(f)
        self.AM_list = self.data['AM_list']
        self.cnn_labels_dict = {value: index for index,
                                value in enumerate(self.data['cnn_dimensions'])}
        # AM_sites_list = []
        # for AM in self.AM_list:
        #     for site in AM['sites']:
        #         AM_sites_list.append(site['label'])
        # self.AM_sites_list = AM_sites_list

    def get_AnonymousMotif(self, AM_label: str):
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

    def get_AM_from_OQMD_id(self, id: int) -> Union[AnonymousMotif, None]:
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
        anonymous_structure = get_anonymous_struct(structure)
        struct_AM_label = get_as_label(anonymous_structure)
        sm = StructureMatcher(stol=0.15)
        for AM in self.AM_list:
            vanilla_label = '_'.join(AM['AM_label'].split('_')[:-1])
            if vanilla_label == struct_AM_label:
                example_structure = Structure.from_str(
                    AM['AM_example_structure']['structure'], fmt='json')
                groups = sm.group_structures(
                    [example_structure, anonymous_structure])
                if len(groups) > 1:
                    return AnonymousMotif(AM)
        return None

    def get_AMSite(self, AM_site_label: str):
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
        AM_label = AM_site_label.split('[')[0]
        for AM in self.AM_list:
            if AM['AM_label'] == AM_label:
                for site in AM['sites']:
                    if site['label'] == AM_site_label:
                        return AMSite(site)
        return None


class RecommenderSystem:
    """
    RecommenderSystem class provides recommendations for ions and sites based on distance in an embedding space.

    This system leverages word embeddings for ions and sites to measure their similarity. Recommendations are made 
    based on these similarity measures, with a certain threshold distance indicating a meaningful recommendation.

    Attributes:
    occupation_data (OccupationData): An instance of OccupationData containing the occupation states of ions and sites.
    distance_threshold (float): The distance threshold for recommendations. Only ions or sites within this threshold 
                                distance are considered as potential recommendations.
    embedding (KeyedVectors): The general word embeddings of ions and sites.
    ions_embedding (KeyedVectors): Word embeddings specifically for ions.
    sites_embedding (KeyedVectors): Word embeddings specifically for sites.
    ion_forbidden_list (list): A list of ions which are not to be recommended.
    """

    def __init__(self, occupation_data: OccupationData,
                 embedding: KeyedVectors,
                 distance_threshold: float,
                 ion_forbidden_list: list):
        """
        Parameters
        ----------
        occupation_data : OccupationData
            The occupation data containing occupation states of ions and sites.
        embedding : KeyedVectors
            The general word embeddings for ions and sites.
        distance_threshold : float
            The distance threshold for recommendations. Only ions or sites within this threshold distance are considered 
            as potential recommendations.
        ion_forbidden_list : list
            A list of ions which are not to be recommended.
        """
        self.occupation_data = occupation_data
        self.distance_threshold = distance_threshold
        self.embedding = embedding
        self.ion_forbidden_list = ion_forbidden_list

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

    def check_recommendation_novelty(self, ion: str, site_label: str) -> bool:
        """
        Check whether a given ion has ever occupied a particular site in an anonymous motif (AM).

        Parameters
        ----------
        ion : str
            The label of the ion to be checked.
        site_label : str
            The label of the site where the ion's presence is being checked.

        Returns
        -------
        bool
            True if the ion has never occupied the site in the anonymous motif (AM), indicating a novel recommendation. False otherwise.
        """
        AM_label = site_label.split('[')[0]
        for AM in self.occupation_data.AM_list:
            if AM['AM_label'] == AM_label:
                for site in AM['sites']:
                    if site['label'] == site_label:
                        if site['ion_distribution'][ion] == 0:
                            return True
                        else:
                            return False

    def get_recommendation_for_ion(self, ion: str, top_n: int = None, only_new: bool = False, local_geometry: tuple = None) -> list[tuple[str, AMSite, bool]]:
        """
        Generate a list of site recommendations for a given ion.

        The method evaluates the similarity between the ion and available sites using embeddings, returning a list 
        of recommended sites. The list is sorted by ascending order of distance, which measures the dissimilarity 
        between the ion and site embeddings.

        Parameters
        ----------
        ion : str
            The ion for which site recommendations are to be generated.
        top_n: int 
            The size limit of the recommendations list.
        only_new: bool
            Only returns recommendation for new ion-site occupations.

        Returns
        -------
        list[tuple[str, AMSite, bool]]
            Each tuple consists of a recommended site label, the distance of the site from the ion in the embedding 
            space, and a boolean indicating whether the recommendation is novel.
        """
        if ion not in self.ions_embedding.index_to_key:
            raise ValueError(f"Ion {ion} not found in the embedding index.")

        recommendation_list = []
        for site in self.sites_embedding.index_to_key:
            dist = self.embedding.distance(ion, site)
            if dist < self.distance_threshold:
                recommendation_novelty = self.check_recommendation_novelty(
                    ion, site)
                recommendation_list.append(
                    (site, dist, recommendation_novelty))

        recommendation_list.sort(key=lambda x: x[1])
        recommendation_list = [(self.occupation_data.get_AMSite(t[0]),
                               t[1],
                               t[2]) for t in recommendation_list]

        recommendation_list_filtered = []
        if local_geometry:
            local_geometry_label, local_geometry_min_value = local_geometry
            local_geometry_index = self.occupation_data.cnn_labels_dict[local_geometry_label]
            for rec in recommendation_list:
                site = rec[0]
                site_CNNf_median = site.CNNf_median
                if site_CNNf_median[local_geometry_index] >= local_geometry_min_value:
                    recommendation_list_filtered.append(rec)
            recommendation_list = recommendation_list_filtered

        if top_n:
            if len(recommendation_list) > top_n:
                recommendation_list = recommendation_list[:top_n]

        if only_new:
            recommendation_list = [t for t in recommendation_list if t[2]]

        return recommendation_list

    def get_recommendation_for_site(self, site: str, top_n: int = None, only_new: bool = False) -> list[tuple[str, float, bool]]:
        """
        Generate a list of ion recommendations for a given site.

        The method evaluates the similarity between the site and available ions using embeddings, returning a list 
        of recommended ions. The list is sorted by ascending order of distance, which measures the dissimilarity 
        between the ion and site embeddings.

        Parameters
        ----------
        site : str
            The site for which ion recommendations are to be generated.
        top_n: int 
            The size limit of the recommendations list.
        only_new: bool
            Only returns recommendation for new ion-site occupations.

        Returns
        -------
        list[tuple[str, float, bool]]
            Each tuple consists of a recommended ion label, the distance of the ion from the site in the embedding 
            space, and a boolean indicating whether the recommendation is novel.
        """
        if site not in self.sites_embedding.index_to_key:
            raise ValueError(f"Site {site} not found in the embedding index.")

        recommendation_list = []
        for ion in self.ions_embedding.index_to_key:
            if ion in self.ion_forbidden_list:
                continue
            dist = self.embedding.distance(site, ion)
            if dist < self.distance_threshold:
                recommendation_novelty = self.check_recommendation_novelty(
                    ion, site)
                recommendation_list.append((ion, dist, recommendation_novelty))

        recommendation_list.sort(key=lambda x: x[1])

        if top_n:
            if len(recommendation_list) > top_n:
                recommendation_list = recommendation_list[:top_n]

        if only_new:
            recommendation_list = [t for t in recommendation_list if t[2]]

        return recommendation_list

    def get_recommendation_for_AM(self, AM: AnonymousMotif) -> dict[AMSite, list[tuple[str, float, bool]]]:
        """
        Generate a dictionary of ion recommendations for each site of an Anonymous Motif (AM).

        The method evaluates the similarity between the sites of the AM and available ions using embeddings, 
        returning a dictionary where keys are AMSites and values are lists of recommended ions for each site. 
        Each list is sorted by ascending order of distance, which measures the dissimilarity between the ion 
        and site embeddings.

        Parameters
        ----------
        AM : AnonymousMotif
            The Anonymous Motif (AM) for which ion recommendations are to be generated for each site.

        Returns
        -------
        dict[AMSite, list[tuple[str, float, bool]]]
            A dictionary where each key-value pair consists of an AMSite and a corresponding list of recommended ions.
            Each list item is a tuple consisting of a recommended ion label, the distance of the ion from the site 
            in the embedding space, and a boolean indicating whether the recommendation is novel.
        """
        recommendations = dict()
        for site in AM.sites:
            recommendations[site] = self.get_recommendation_for_site(
                site.label)
        return recommendations
