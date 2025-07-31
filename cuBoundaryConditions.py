def impose(mesh):
    """
    Apply the boundary conditions to the mesh values
    """
    #Periodic case, i.e. no ghost cells defined
    if mesh["GhostCells"].shape[0] == 0:
        return

    #Otherwise, impose the required condition j: element, k: neighbour
    j = mesh["GhostCells"][0,:]
    k = mesh["GhostCells"][1,:]
    s = mesh["GhostCells"][2,:]

    mesh["Wj"][j] = mesh["Wj"][k]
    mesh["HUj"][j] = s*mesh["HUj"][k]
    mesh["HVj"][j] = s*mesh["HVj"][k]

    return
