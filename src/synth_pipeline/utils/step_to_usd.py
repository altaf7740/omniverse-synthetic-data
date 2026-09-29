"""
Utility for converting a STEP/IGES CAD file to USD via Isaac Sim's bundled
Hoops CAD converter (omni.kit.converter.cad / hoops_core).

Requires a SimulationApp to already be running, with the
"omni.kit.converter.cad" and "omni.kit.converter.hoops_core" extensions
enabled, before calling this function - it only does the conversion itself,
not app/extension bootstrapping.
"""

from pathlib import Path


async def convert_step_to_usd(step_path: str, usd_path: str = None) -> str:
    """Convert one STEP/IGES file to USD.

    Args:
        step_path: Full path to the input .step/.stp file.
        usd_path: Full path for the output .usd file. Defaults to the same
            directory and stem as step_path, with a .usd extension.

    Returns:
        The output USD path.

    Raises:
        RuntimeError: If the conversion fails.
    """
    from omni.kit.converter.hoops_core.impl.helper import HoopsConverterHelper

    if usd_path is None:
        usd_path = str(Path(step_path).with_suffix(".usd"))

    helper = HoopsConverterHelper()
    output_url, status = await helper.create_import_task(step_path, usd_path, {})
    helper.destroy()

    if not output_url:
        raise RuntimeError(f"STEP->USD conversion failed for {step_path}: {status}")

    return output_url
