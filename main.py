import networkx as nx
import numpy as np
import random
import time
from collections import defaultdict

# --- Constants and Parameters ---
NUM_NODES = 20  # NY-20 topology
NUM_AIGC_SERVERS = 4
NUM_PEPS_RANGE = (1, 4)
PES_CAPACITIES = [1000, 1250, 1500, 1750, 2000]  # CPU cores
MAX_PROMPT_TYPES_PER_PES = 4
LINK_BANDWIDTHS = [10, 20, 30, 40]  # Gbps
LINK_WEIGHTS = (5, 20)  # Range for link weights
PROMPT_CHAIN_LENGTHS = (3, 7)  # 3 to 7 prompts
PROMPT_COMP_DEMAND = (0.5, 0.8)  # CPU cores
VPEP_COMP_DEMAND = 0.6  # CPU cores
BANDWIDTH_DEMAND = [1, 2, 3, 4, 5]  # Mbps
COMP_LATENCY_PER_CORE = 0.01  # ms/CPU core
PEP_VERIFY_LATENCY = 0.5  # ms
VPEP_VERIFY_LATENCY = 0.02  # ms/Mbps
NUM_NETWORKS = 5
NUM_AI_SRS_PER_NETWORK = 4000

# --- Helper Functions ---
def generate_network():
    # Create a random graph (approximating NY-20 topology)
    G = nx.erdos_renyi_graph(NUM_NODES, 0.3, directed=True)
    
    # Assign node types: S (PESes), Q (PEPs), X (AIGC servers)
    nodes = list(G.nodes)
    random.shuffle(nodes)
    num_peps = random.randint(*NUM_PEPS_RANGE)
    X = nodes[:NUM_AIGC_SERVERS]  # AIGC servers
    Q = nodes[NUM_AIGC_SERVERS:NUM_AIGC_SERVERS + num_peps]  # PEPs
    S = nodes[NUM_AIGC_SERVERS + num_peps:]  # PESes
    
    # Assign attributes to nodes
    for node in G.nodes:
        if node in S:
            G.nodes[node]['type'] = 'PES'
            G.nodes[node]['capacity'] = random.choice(PES_CAPACITIES)
            G.nodes[node]['prompt_types'] = random.sample(range(1, 8), MAX_PROMPT_TYPES_PER_PES)
        elif node in Q:
            G.nodes[node]['type'] = 'PEP'
        else:
            G.nodes[node]['type'] = 'AIGC'
    
    # Assign attributes to edges
    for u, v in G.edges():
        G.edges[u, v]['bandwidth'] = random.choice(LINK_BANDWIDTHS)  # Gbps
        G.edges[u, v]['weight'] = random.uniform(*LINK_WEIGHTS)
    
    return G, S, Q, X

def generate_ai_sr(G, S):
    # Generate an AI-SR with a prompt chain
    chain_length = random.randint(*PROMPT_CHAIN_LENGTHS)
    PrC = list(range(chain_length))  # Prompts p_0, p_1, ..., p_{chain_length-1}
    d_v = [2 for _ in PrC]  # Data volume: 2 Mbps for simplicity
    BW = random.choice(BANDWIDTH_DEMAND)  # Bandwidth demand
    prompt_types = [random.randint(1, 7) for _ in PrC]  # Assign prompt types
    c_v = [random.uniform(*PROMPT_COMP_DEMAND) for _ in PrC]  # Computational demand
    
    # Select source node (from S)
    source = random.choice(S)
    return {'PrC': PrC, 'd_v': d_v, 'BW': BW, 'prompt_types': prompt_types, 'c_v': c_v, 'source': source}

def compute_transmission_latency(G, path, d_v, BW):
    # Compute transmission latency along a path
    if not path:
        return 0
    total_latency = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if (u, v) in G.edges():
            bw = G.edges[u, v]['bandwidth']  # Gbps
            # Convert Gbps to Mbps for consistency: 1 Gbps = 1000 Mbps
            # Latency = data volume / bandwidth (ms), approximated
            latency = (d_v / (bw * 1000)) * 1000  # Convert to ms
            total_latency += latency
    return total_latency

# --- SCBB-L Algorithm Implementation ---
def scbb_l(G, S, Q, X, ai_sr):
    PrC = ai_sr['PrC']
    d_v = ai_sr['d_v']
    BW = ai_sr['BW']
    prompt_types = ai_sr['prompt_types']
    c_v = ai_sr['c_v']
    source = ai_sr['source']
    
    # Track remaining capacity of PESes
    remaining_capacity = {n: G.nodes[n]['capacity'] for n in S}
    
    # Initialize forwarding path
    For_Path = [source]
    i = 1  # Start after p_0 (source)
    n_last = source
    PrC = PrC[1:]  # Remove p_0
    d_v = d_v[1:]
    prompt_types = prompt_types[1:]
    c_v = c_v[1:]
    
    # Track deployed prompts and their locations
    deployment = {}
    
    while i < len(PrC) + 1:
        temp_k = []
        temp_T = float('inf')
        Sub_Path = []
        best_n = None
        
        # Try subsequences of length m starting at index i
        for m in range(1, len(PrC) - i + 2):
            kappa = PrC[i-1:i-1+m]
            kappa_types = prompt_types[i-1:i-1+m]
            kappa_c_v = c_v[i-1:i-1+m]
            
            for n in S:
                # Check if PES n can host all prompts in kappa (prompt types and capacity)
                can_host = all(pt in G.nodes[n]['prompt_types'] for pt in kappa_types)
                total_comp = sum(kappa_c_v)
                if n != n_last:
                    total_comp += VPEP_COMP_DEMAND  # For vPEP if needed
                if not can_host or total_comp > remaining_capacity[n]:
                    continue
                
                # Compute T_add
                if n == n_last:
                    T_add = sum(c * COMP_LATENCY_PER_CORE for c in kappa_c_v)
                else:
                    # Compute PEP-based verification latency
                    pep_latencies = []
                    for q in Q:
                        try:
                            path_to_q = nx.shortest_path(G, n_last, q, weight='weight')
                            path_from_q = nx.shortest_path(G, q, n, weight='weight')
                            trans_to_q = compute_transmission_latency(G, path_to_q, d_v[i-2], BW)
                            trans_from_q = compute_transmission_latency(G, path_from_q, d_v[i-2], BW)
                            pep_latency = trans_to_q + trans_from_q + PEP_VERIFY_LATENCY
                            pep_latencies.append(pep_latency)
                        except nx.NetworkXNoPath:
                            continue
                    
                    # Compute vPEP-based verification latency
                    try:
                        path_direct = nx.shortest_path(G, n_last, n, weight='weight')
                        trans_direct = compute_transmission_latency(G, path_direct, d_v[i-2], BW)
                        vpep_latency = VPEP_VERIFY_LATENCY * d_v[i-2] + trans_direct
                    except nx.NetworkXNoPath:
                        vpep_latency = float('inf')
                    
                    # Take the minimum verification latency
                    min_verify_latency = min(pep_latencies + [vpep_latency]) if pep_latencies else vpep_latency
                    T_add = min_verify_latency + sum(c * COMP_LATENCY_PER_CORE for c in kappa_c_v)
                
                # Compute VLB
                VLB = T_add / m
                if VLB < temp_T:
                    temp_T = VLB
                    temp_k = kappa
                    best_n = n
                    if n == n_last:
                        Sub_Path = [n]
                    else:
                        if min_verify_latency == vpep_latency:
                            Sub_Path = path_direct
                        else:
                            # Find the PEP that gave the minimum latency
                            for q in Q:
                                path_to_q = nx.shortest_path(G, n_last, q, weight='weight')
                                path_from_q = nx.shortest_path(G, q, n, weight='weight')
                                trans_to_q = compute_transmission_latency(G, path_to_q, d_v[i-2], BW)
                                trans_from_q = compute_transmission_latency(G, path_from_q, d_v[i-2], BW)
                                if trans_to_q + trans_from_q + PEP_VERIFY_LATENCY == min_verify_latency:
                                    Sub_Path = path_to_q[:-1] + path_from_q
                                    break
        
        if not temp_k:  # No feasible deployment found
            return None, float('inf'), {}
        
        # Update deployment and resources
        for idx, p in enumerate(temp_k):
            deployment[p] = best_n
            remaining_capacity[best_n] -= c_v[i-1+idx]
        if best_n != n_last:
            remaining_capacity[n_last] -= VPEP_COMP_DEMAND  # For vPEP
        
        i += len(temp_k)
        For_Path.extend(Sub_Path[1:] if Sub_Path[0] == n_last else Sub_Path)
        n_last = best_n
    
    # Connect to AIGC server
    min_latency = float('inf')
    best_x = None
    final_path = []
    for x in X:
        try:
            # Try vPEP
            path_direct = nx.shortest_path(G, n_last, x, weight='weight')
            trans_direct = compute_transmission_latency(G, path_direct, d_v[-1], BW)
            vpep_latency = VPEP_VERIFY_LATENCY * d_v[-1] + trans_direct
            if vpep_latency < min_latency:
                min_latency = vpep_latency
                best_x = x
                final_path = path_direct
            
            # Try PEP
            for q in Q:
                path_to_q = nx.shortest_path(G, n_last, q, weight='weight')
                path_from_q = nx.shortest_path(G, q, x, weight='weight')
                trans_to_q = compute_transmission_latency(G, path_to_q, d_v[-1], BW)
                trans_from_q = compute_transmission_latency(G, path_from_q, d_v[-1], BW)
                pep_latency = trans_to_q + trans_from_q + PEP_VERIFY_LATENCY
                if pep_latency < min_latency:
                    min_latency = pep_latency
                    best_x = x
                    final_path = path_to_q[:-1] + path_from_q
        except nx.NetworkXNoPath:
            continue
    
    if best_x is None:
        return None, float('inf'), deployment
    
    For_Path.extend(final_path[1:] if final_path[0] == n_last else final_path)
    deployment[PrC[-1]] = best_x  # Last prompt goes to AIGC server
    
    # Compute total latency
    total_latency = 0
    for idx, p in enumerate(PrC):
        n = deployment[p]
        # Computation latency
        total_latency += c_v[idx] * COMP_LATENCY_PER_CORE
        # Transmission and verification latency
        if idx == 0:
            continue
        prev_n = deployment[PrC[idx-1]]
        if n != prev_n:
            # Find the path used
            path = []
            for j in range(len(For_Path)):
                if For_Path[j] == prev_n:
                    for k in range(j, len(For_Path)):
                        path.append(For_Path[k])
                        if For_Path[k] == n:
                            break
                    break
            trans_latency = compute_transmission_latency(G, path, d_v[idx-1], BW)
            # Check if path goes through a PEP
            uses_pep = any(node in Q for node in path[1:-1])
            if uses_pep:
                verify_latency = PEP_VERIFY_LATENCY
            else:
                verify_latency = VPEP_VERIFY_LATENCY * d_v[idx-1]
            total_latency += trans_latency + verify_latency
    
    return For_Path, total_latency, deployment

# --- PCDO Implementation (No ZT Verification) ---
def pcdo(G, S, X, ai_sr):
    PrC = ai_sr['PrC']
    d_v = ai_sr['d_v']
    BW = ai_sr['BW']
    prompt_types = ai_sr['prompt_types']
    c_v = ai_sr['c_v']
    source = ai_sr['source']
    
    remaining_capacity = {n: G.nodes[n]['capacity'] for n in S}
    deployment = {}
    For_Path = [source]
    PrC = PrC[1:]  # Remove p_0
    d_v = d_v[1:]
    prompt_types = prompt_types[1:]
    c_v = c_v[1:]
    
    # Greedy deployment: assign each prompt to the closest PES with capacity
    current = source
    for idx, p in enumerate(PrC):
        pt = prompt_types[idx]
        best_n = None
        min_dist = float('inf')
        for n in S:
            if pt not in G.nodes[n]['prompt_types']:
                continue
            if c_v[idx] > remaining_capacity[n]:
                continue
            try:
                dist = nx.shortest_path_length(G, current, n, weight='weight')
                if dist < min_dist:
                    min_dist = dist
                    best_n = n
            except nx.NetworkXNoPath:
                continue
        
        if best_n is None:
            return None, float('inf'), deployment
        
        deployment[p] = best_n
        remaining_capacity[best_n] -= c_v[idx]
        path = nx.shortest_path(G, current, best_n, weight='weight')
        For_Path.extend(path[1:] if path[0] == current else path)
        current = best_n
    
    # Connect to AIGC server
    min_dist = float('inf')
    best_x = None
    final_path = []
    for x in X:
        try:
            dist = nx.shortest_path_length(G, current, x, weight='weight')
            if dist < min_dist:
                min_dist = dist
                best_x = x
                final_path = nx.shortest_path(G, current, x, weight='weight')
        except nx.NetworkXNoPath:
            continue
    
    if best_x is None:
        return None, float('inf'), deployment
    
    For_Path.extend(final_path[1:] if final_path[0] == current else final_path)
    deployment[PrC[-1]] = best_x
    
    # Compute total latency (no verification)
    total_latency = 0
    for idx, p in enumerate(PrC):
        total_latency += c_v[idx] * COMP_LATENCY_PER_CORE
        if idx == 0:
            continue
        prev_n = deployment[PrC[idx-1]]
        n = deployment[p]
        if n != prev_n:
            path = nx.shortest_path(G, prev_n, n, weight='weight')
            trans_latency = compute_transmission_latency(G, path, d_v[idx-1], BW)
            total_latency += trans_latency
    
    return For_Path, total_latency, deployment

# --- PCDF Implementation (PCDO + Least-Latency Verification) ---
def pcdf(G, S, Q, X, ai_sr):
    # First run PCDO to get the deployment path and deployment
    For_Path, _, deployment = pcdo(G, S, X, ai_sr)
    if For_Path is None:
        return None, float('inf')
    
    PrC = ai_sr['PrC'][1:]
    d_v = ai_sr['d_v'][1:]
    BW = ai_sr['BW']
    c_v = ai_sr['c_v'][1:]
    
    # Add verification: for each inter-PES transition, choose PEP or vPEP
    new_path = [For_Path[0]]
    total_latency = 0
    for idx, p in enumerate(PrC):
        total_latency += c_v[idx] * COMP_LATENCY_PER_CORE
        if idx == 0:
            continue
        prev_n = deployment[PrC[idx-1]]
        n = deployment[p]
        if n != prev_n:
            # Choose between PEP and vPEP
            pep_latencies = []
            for q in Q:
                try:
                    path_to_q = nx.shortest_path(G, prev_n, q, weight='weight')
                    path_from_q = nx.shortest_path(G, q, n, weight='weight')
                    trans_to_q = compute_transmission_latency(G, path_to_q, d_v[idx-1], BW)
                    trans_from_q = compute_transmission_latency(G, path_from_q, d_v[idx-1], BW)
                    pep_latency = trans_to_q + trans_from_q + PEP_VERIFY_LATENCY
                    pep_latencies.append((pep_latency, path_to_q[:-1] + path_from_q))
                except nx.NetworkXNoPath:
                    continue
            
            path_direct = nx.shortest_path(G, prev_n, n, weight='weight')
            trans_direct = compute_transmission_latency(G, path_direct, d_v[idx-1], BW)
            vpep_latency = VPEP_VERIFY_LATENCY * d_v[idx-1] + trans_direct
            
            if pep_latencies and min(pep_latencies, key=lambda x: x[0])[0] < vpep_latency:
                latency, path = min(pep_latencies, key=lambda x: x[0])
            else:
                latency = vpep_latency
                path = path_direct
            
            total_latency += latency
            new_path.extend(path[1:] if path[0] == new_path[-1] else path)
    
    # Connect to AIGC server (already handled by PCDO, just recompute latency)
    current = new_path[-1]
    for x in X:
        if x in For_Path:
            path = nx.shortest_path(G, current, x, weight='weight')
            trans_direct = compute_transmission_latency(G, path, d_v[-1], BW)
            vpep_latency = VPEP_VERIFY_LATENCY * d_v[-1] + trans_direct
            total_latency += vpep_latency
            break
    
    return new_path, total_latency

# --- Main Experiment ---
def run_experiment():
    results = {'SCBB-L': {'latency': [], 'accepted': 0, 'runtime': []},
               'PCDO': {'latency': [], 'accepted': 0, 'runtime': []},
               'PCDF': {'latency': [], 'accepted': 0, 'runtime': []}}
    
    for network_idx in range(NUM_NETWORKS):
        print(f"Processing Network {network_idx + 1}/{NUM_NETWORKS}...")
        G, S, Q, X = generate_network()
        accepted = {'SCBB-L': 0, 'PCDO': 0, 'PCDF': 0}
        latencies = {'SCBB-L': [], 'PCDO': [], 'PCDF': []}
        runtimes = {'SCBB-L': [], 'PCDO': [], 'PCDF': []}
        
        for sr_idx in range(NUM_AI_SRS_PER_NETWORK):
            if (sr_idx + 1) % 500 == 0:  # Update every 500 AI-SRs
                print(f"  Network {network_idx + 1}: Processed {sr_idx + 1}/{NUM_AI_SRS_PER_NETWORK} AI-SRs")
            ai_sr = generate_ai_sr(G, S)
            
            # Run SCBB-L
            start_time = time.time()
            path, latency, _ = scbb_l(G, S, Q, X, ai_sr)
            runtimes['SCBB-L'].append(time.time() - start_time)
            if path is not None:
                accepted['SCBB-L'] += 1
                latencies['SCBB-L'].append(latency)
            
            # Run PCDO
            start_time = time.time()
            path, latency, _ = pcdo(G, S, X, ai_sr)
            runtimes['PCDO'].append(time.time() - start_time)
            if path is not None:
                accepted['PCDO'] += 1
                latencies['PCDO'].append(latency)
            
            # Run PCDF
            start_time = time.time()
            path, latency = pcdf(G, S, Q, X, ai_sr)
            runtimes['PCDF'].append(time.time() - start_time)
            if path is not None:
                accepted['PCDF'] += 1
                latencies['PCDF'].append(latency)
        
        # Aggregate results for this network
        for method in results:
            results[method]['accepted'] += accepted[method]
            results[method]['latency'].extend(latencies[method])
            results[method]['runtime'].extend(runtimes[method])
    
    # Compute final metrics
    print("\nFinal Results:")
    for method in results:
        total_requests = NUM_NETWORKS * NUM_AI_SRS_PER_NETWORK
        acceptance_ratio = results[method]['accepted'] / total_requests
        avg_latency = np.mean(results[method]['latency']) if results[method]['latency'] else float('inf')
        avg_runtime = np.mean(results[method]['runtime']) * 1000  # Convert to ms
        print(f"{method}:")
        print(f"  Acceptance Ratio: {acceptance_ratio:.4f}")
        print(f"  Average Latency: {avg_latency:.4f} ms")
        print(f"  Average Runtime: {avg_runtime:.4f} ms")

if __name__ == "__main__":
    run_experiment()