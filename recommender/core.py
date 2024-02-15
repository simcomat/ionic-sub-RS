from __future__ import annotations
from typing import Union
from pymatgen.core import Structure, Element, Composition, periodic_table
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

    def __eq__(self,another: AMSite):
        """
        Checks if another AMsite is the same as self.

        Returns
        -------
        bool
            True if AMsites are the same, False otherwise.
        """
        return self.data == another.data

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
            recommendations[site.label] = self.get_recommendation_for_site(
                site.label)
        return recommendations


    def get_recommendation_for_chemsys(self, elements: list, top_n: int = 500, ignore_above_n: int = 300) -> dict[str, list[tuple[str, float, bool]], str, list, dict, float, float]:
        """
        Generates a dictionary of recommended compounds in the specified chemical system.

        The method starts by generating recommendations for the atoms in the chemical system, then checks which Anonymous Motifs
        got recommended for every atom in the chemical system, finally selecting Anonymous Motifs whose AMSites were all recommended.
        The method returns a dictionary of the recommended compounds and information about them.

        Parameters
        ----------
        elements : list
            The chemical elements defining the chemical space.
        top_n : int
            The number of top recommendations for each atom.
        ignore_above_n : int
            Threshold to ignore Anonymous Motifs that give rise to too many compounds. 

        Returns
        -------
        dict[str, list[tuple[str, float, bool]], str, list, dict, float, float]
            A dictionary carrying information about each recommended compound: formula, Anonymous Motif, Anonymous Motif label,
            list of atom recommendation for each wickoff position of the Anonymous Motif, list of dictionaries with oxidation
            states of the compound, sum of all the atom-site distances, and novelty (site-atom occupancy) fraction.
        """
        elements=list(set(elements))
        assert all([ periodic_table.ElementBase.is_valid_symbol(el) for el in elements]), "One or more of the input elements are not valid."

        # Gets top_n recommendations for each atomic element in compound
        recommendations_atoms = { atom:self.get_recommendation_for_ion(atom, top_n=top_n, only_new=False) for atom in elements }

        # Anonymous Motifs recommended simultaneously for all the elements
        AMlabels = { item[0]:[ rec[0].AM_label for rec in item[1] ] for item in recommendations_atoms.items() }
        AMs_intersect = list(set.intersection(*map(set,AMlabels.values())))

        # Dictionary that will collect the recommended compounds
        recommendations_compounds={'formula':[],'AM':[],'AM_label':[],'atom_AMsites':[],'oxidation_states':[],'sum_SiteAtom_dists':[],'novelty_fraction':[]}

        # Loops over the labels (corresponding to Anonymous Motifs for which there are recommendations for all the atoms in the compound
        for label in AMs_intersect:
            AM=self.occupation_data.get_AnonymousMotif(label)

            AMsites={}
            rec_AMsites={}
            for k in AMlabels.keys():
                AMsites[k]=[ recommendations_atoms[k][i] for i, lab in enumerate(AMlabels[k]) if lab==label ]
                rec_AMsites[k]=[]
                rec_AMsites[k+'_dists']=[]

            for i, site in enumerate(AM.sites): # Loops over the AMSites in the current AM
                for k in AMsites.keys(): # Loops over each atom in the compound
                    if any([ amsite[0]==site for amsite in AMsites[k] ]):
                        rec_AMsites[k].append(i)
            total_recs=list(set(sum([rec_AMsites[k] for k in rec_AMsites.keys()],[])))
            if len(total_recs) == len(AM.sites):
                inv_dict = {site_idx:[] for site_idx in range(len(AM.sites))} # Dictionary of recommended atoms for each AMSite in the current AM
                for atom, site_indices in rec_AMsites.items():
                    for site_idx in site_indices:
                        inv_dict[site_idx].append(atom)
                # Avoiding the use of itertools
                compounds = [[]]
                for proposed_atoms in [atom for site_idx, atom in inv_dict.items()]:
                    compounds = [ x+[y] for x in compounds for y in proposed_atoms ]
                if len(compounds)>ignore_above_n: # Ignores AM resulting in too many compounds
                    continue

                # Collects information on each recommended compound
                for compound in compounds:
                    formula, oxi_states=list_to_formula(compound, AM) # Reduced formula and oxidation states
                    if not all([ e in formula for e in elements]): # Ignores compounds lacking at least one instance of elements
                        continue
                    # Total atom-site distance and novelty fraction
                    total_dist=0
                    novelty_fraction=0
                    for ix,atom in enumerate(compound):
                        site=(AM.sites)[ix]
                        total_dist+=sum([ rec[1] for rec in recommendations_atoms[atom] if site==rec[0] ])
                        novelty_fraction+=sum([ rec[2] for rec in recommendations_atoms[atom] if site==rec[0] ])
                    novelty_fraction/=len(AM.sites)

                    recommendations_compounds['formula'].append(formula)
                    recommendations_compounds['AM'].append(AM)
                    recommendations_compounds['AM_label'].append(label)
                    recommendations_compounds['atom_AMsites'].append(compound)
                    recommendations_compounds['oxidation_states'].append(oxi_states)
                    recommendations_compounds['sum_SiteAtom_dists'].append(total_dist)
                    recommendations_compounds['novelty_fraction'].append(novelty_fraction)

        return recommendations_compounds


    def get_recommendation_for_compound(self, formula: str, top_n: int = 500) -> dict[AMSite, list[tuple[str, float, bool]]]:
        """
        Generates a dictionary of recommended compounds with the specified formula.

        The method starts by generating recommendations of compounds in a chemical system, and then selects only the ones matching
        the specified chemical formula.

        Parameters
        ----------
        formula : str
            The chemical formula for which we want recommendations.
        top_n : int
            The number of top recommendations for each atom.

        Returns
        -------
        dict[str, list[tuple[str, float, bool]], str, list, dict, float, float]
            A dictionary carrying information about each recommended compound: formula, Anonymous Motif, Anonymous Motif label,
            list of atom recommendation for each wickoff position of the Anonymous Motif, list of dictionaries with oxidation
            states of the compound, sum of all the atom-site distances, and novelty (site-atom occupancy) fraction.
        """
        comp = Composition(formula)
        elements = comp.chemical_system.split('-')
        recommended_compounds_in_chemsys = self.get_recommendation_for_chemsys(elements, top_n=top_n)
        reduced_formula = Composition(formula).reduced_formula
        matches_idx = [ i for i, compound in enumerate(recommended_compounds_in_chemsys['formula']) if compound==reduced_formula ]
        recommendations = { key:[values[i] for i in matches_idx] for key, values in recommended_compounds_in_chemsys.items() }

        return recommendations


def list_to_formula(compound: list, AM: AnonymousMotif) -> tuple[str, dict]:
    """
    Transforms a list of atoms (eg. ['A','B','B']) defining the occupation of the AMSites [1,2,3] of the AM given as input, into a
    chemical formula for that compound (counts the multiplicity of each AMSite), eg. 'A3B4'.
    """
    # Counts occurrencies of each species
    atom_count=[ [atom]*((AM.sites)[site_idx].site_sym[0]) for site_idx, atom in enumerate(compound) ]
    atom_count=sum(atom_count,[])
    occurrencies={atom:atom_count.count(atom) for atom in set(atom_count)} # This eliminates any atom with 0 count
    comp=Composition(occurrencies)
    return comp.reduced_formula, list(comp.oxi_state_guesses())

