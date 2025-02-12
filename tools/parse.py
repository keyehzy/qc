#!/usr/bin/env python3

import argparse
import json
import sys

# A dictionary mapping atomic number strings to element symbols.
ATOMIC_NUMBER_TO_SYMBOL = {
    "1": "H",
    "2": "He",
    "3": "Li",
    "4": "Be",
    "5": "B",
    "6": "C",
    "7": "N",
    "8": "O",
    "9": "F",
    "10": "Ne",
    "11": "Na",
    "12": "Mg",
    "13": "Al",
    "14": "Si",
    "15": "P",
    "16": "S",
    "17": "Cl",
    "18": "Ar",
    "19": "K",
    "20": "Ca",
    "21": "Sc",
    "22": "Ti",
    "23": "V",
    "24": "Cr",
    "25": "Mn",
    "26": "Fe",
    "27": "Co",
    "28": "Ni",
    "29": "Cu",
    "30": "Zn",
    "31": "Ga",
    "32": "Ge",
    "33": "As",
    "34": "Se",
    "35": "Br",
    "36": "Kr",
    "37": "Rb",
    "38": "Sr",
    "39": "Y",
    "40": "Zr",
    "41": "Nb",
    "42": "Mo",
    "43": "Tc",
    "44": "Ru",
    "45": "Rh",
    "46": "Pd",
    "47": "Ag",
    "48": "Cd",
    "49": "In",
    "50": "Sn",
    "51": "Sb",
    "52": "Te",
    "53": "I",
    "54": "Xe",
    "55": "Cs",
    "56": "Ba",
    "57": "La",
    "58": "Ce",
    "59": "Pr",
    "60": "Nd",
    "61": "Pm",
    "62": "Sm",
    "63": "Eu",
    "64": "Gd",
    "65": "Tb",
    "66": "Dy",
    "67": "Ho",
    "68": "Er",
    "69": "Tm",
    "70": "Yb",
    "71": "Lu",
    "72": "Hf",
    "73": "Ta",
    "74": "W",
    "75": "Re",
    "76": "Os",
    "77": "Ir",
    "78": "Pt",
    "79": "Au",
    "80": "Hg",
    "81": "Tl",
    "82": "Pb",
    "83": "Bi",
    "84": "Po",
    "85": "At",
    "86": "Rn",
    "87": "Fr",
    "88": "Ra",
    "89": "Ac",
    "90": "Th",
    "91": "Pa",
    "92": "U",
    "93": "Np",
    "94": "Pu",
    "95": "Am",
    "96": "Cm",
    "97": "Bk",
    "98": "Cf",
    "99": "Es",
    "100": "Fm",
    "101": "Md",
    "102": "No",
    "103": "Lr",
    "104": "Rf",
    "105": "Db",
    "106": "Sg",
    "107": "Bh",
    "108": "Hs",
    "109": "Mt",
    "110": "Ds",
    "111": "Rg",
    "112": "Cn",
    "113": "Nh",
    "114": "Fl",
    "115": "Mc",
    "116": "Lv",
    "117": "Ts",
    "118": "Og"
}

def format_float(number_str):
    """Format a number string to a C++ float literal with a trailing 'f'.

    Args:
        number_str (str): A string representing a number (e.g., "0.3425250914E+01").

    Returns:
        str: The formatted float literal (e.g., "0.3425250914E+01f").
    """
    return number_str + "f"


def format_list(values, is_float=False):
    """Format a list of values as a C++ initializer list.

    Args:
        values (list): List of values (strings or ints).
        is_float (bool): If True, format each value as a float literal.

    Returns:
        str: A string in the form "{value1, value2, ...}".
    """
    formatted_items = []
    for item in values:
        if is_float:
            formatted_items.append(format_float(item))
        else:
            formatted_items.append(str(item))
    return "{" + ", ".join(formatted_items) + "}"


def generate_electron_shell(shell):
    """Generate the C++ initializer for an ElectronShell from a JSON shell.

    Args:
        shell (dict): JSON object representing an electron shell.

    Returns:
        str: A string containing the C++ initializer for the ElectronShell.
    """
    # Format the angular momentum list (assumed to be a list of ints).
    ang_mom = shell.get("angular_momentum", [])
    ang_mom_str = format_list(ang_mom, is_float=False)

    # Format the exponents list (list of strings representing floats).
    exponents = shell.get("exponents", [])
    exponents_str = format_list(exponents, is_float=True)

    # Format the coefficients, which is a list of lists.
    coeffs = shell.get("coefficients", [])
    coeffs_str_list = []
    for coeff_group in coeffs:
        coeff_group_str = format_list(coeff_group, is_float=True)
        coeffs_str_list.append(coeff_group_str)
    # Create the initializer for the coefficients vector-of-vectors.
    coefficients_str = "{" + ", ".join(coeffs_str_list) + "}"

    # Return the full initializer for ElectronShell.
    return f"ElectronShell{{ {ang_mom_str}, {exponents_str}, {coefficients_str} }}"


def generate_basis_set(basis_data):
    """Generate the C++ static BasisSet object from the JSON data.

    Args:
        basis_data (dict): The JSON dictionary representing the basis set.

    Returns:
        str: The generated C++ code as a string.
    """
    # Use the "name" field from JSON as the basis set name;
    # replace hyphens with underscores for a valid C++ identifier.
    basis_name = basis_data.get("name", "BASIS_SET")
    cpp_var_name = basis_name.replace("-", "_")

    # Get the elements data.
    elements = basis_data.get("elements", {})

    # Build the C++ initializer for each element.
    element_entries = []
    for atomic_num, element_data in elements.items():
        # Convert atomic number (as string) to element symbol.
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_num, atomic_num)
        # Each element has one or more electron shells.
        shells = element_data.get("electron_shells", [])
        shell_entries = []
        for shell in shells:
            shell_entries.append(generate_electron_shell(shell))
        # Create the initializer list for the shells.
        shells_str = "{ " + ", ".join(shell_entries) + " }"
        element_entry = f'    {{"{symbol}", {shells_str}}}'
        element_entries.append(element_entry)

    # Combine all element entries.
    elements_str = ",\n".join(element_entries)

    # Construct the final C++ static object code.
    cpp_code = (
        f"static BasisSet {cpp_var_name} {{\n"
        "  {\n"
        f"{elements_str}\n"
        "  }\n"
        "};"
    )
    return cpp_code


def main():
    """Parse the JSON file and print the C++ static object."""
    parser = argparse.ArgumentParser(
        description="Convert a JSON basis set file into a C++ static object."
    )
    parser.add_argument(
        "json_file",
        help="Path to the JSON file containing the basis set data."
    )
    args = parser.parse_args()

    try:
        with open(args.json_file, "r") as file_handle:
            basis_data = json.load(file_handle)
    except Exception as err:
        sys.stderr.write(f"Error reading JSON file: {err}\n")
        sys.exit(1)

    cpp_code = generate_basis_set(basis_data)
    print(cpp_code)


if __name__ == "__main__":
    main()
