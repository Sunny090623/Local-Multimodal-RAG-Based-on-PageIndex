"""
Shared utility functions for the backend.

Consolidates helpers that were previously duplicated across app.py and rag.py.
"""

from pageindex.utils import sanitize_filename  # re-export for convenience


def find_node_by_id(structure, node_id):
    """Recursively search a document structure tree for a node with the given node_id."""
    if isinstance(structure, list):
        for item in structure:
            res = find_node_by_id(item, node_id)
            if res:
                return res
    elif isinstance(structure, dict):
        if structure.get('node_id') == node_id:
            return structure
        if 'nodes' in structure:
            res = find_node_by_id(structure['nodes'], node_id)
            if res:
                return res
    return None


def map_structure_keys(structure):
    """
    Recursively maps structure list/dict keys:
    - Add 'children' (as alias to 'nodes')
    - Add 'page' (as alias to 'start_index')
    """
    if isinstance(structure, list):
        return [map_structure_keys(item) for item in structure]
    elif isinstance(structure, dict):
        mapped = dict(structure)
        if 'nodes' in mapped:
            mapped['children'] = map_structure_keys(mapped['nodes'])
        if 'start_index' in mapped:
            mapped['page'] = mapped['start_index']
        # Also recursively map standard children if any
        if 'children' in mapped:
            mapped['children'] = map_structure_keys(mapped['children'])
        return mapped
    return structure
