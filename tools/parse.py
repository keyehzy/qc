#!/usr/bin/env python3

import argparse
import json
import sys

def format_list(values):
    formatted_items = []
    for item in values:
        formatted_items.append(str(item))
    return "{" + ", ".join(formatted_items) + "}"

def generate_electron_shell(shell):
    ang_mom = shell.get("angular_momentum", [])
    ang_mom_str = format_list(ang_mom)

    exponents = shell.get("exponents", [])
    exponents_str = format_list(exponents)

    coeffs = shell.get("coefficients", [])
    coeffs_str_list = []

    for coeff_group in coeffs:
        coeff_group_str = format_list(coeff_group)
        coeffs_str_list.append(coeff_group_str)

    coefficients_str = "{" + ", ".join(coeffs_str_list) + "}"
    return f"Shell{{ {ang_mom_str}, {exponents_str}, {coefficients_str} }}"

def generate_basis_set(basis_data):
    basis_name = basis_data.get("name", "BASIS_SET")
    cpp_var_name = basis_name.replace("-", "_") .replace(" ", "_") .replace("(", "_") .replace(")", "_") .replace(",", "_") .replace("+", "_plus_") .replace("*","_star_")

    elements = basis_data.get("elements", {})

    element_entries = []
    for atomic_num, element_data in elements.items():
        shells = element_data.get("electron_shells", [])
        shell_entries = []
        for shell in shells:
            shell_entries.append(generate_electron_shell(shell))
        shells_str = "{ " + ", ".join(shell_entries) + " }"
        element_entry = f'    {{{atomic_num}, {shells_str}}}'
        element_entries.append(element_entry)

    elements_str = ",\n".join(element_entries)

    cpp_code = (
        "#pragma once\n\n"
        f"static BasisSet BS_{cpp_var_name} {{\n"
        "  {\n"
        f"{elements_str}\n"
        "  }\n"
        "};"
    )
    return cpp_code

def main():
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
