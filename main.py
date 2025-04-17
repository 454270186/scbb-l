import networkx as nx
import numpy as np
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt

# --- Constants and Parameters ---
NUM_NODES = 20
NUM_AIGC_SERVERS = 4
NUM_PEPS_RANGE = (1, 4)
PES_CAPACITIES = [1000, 1250, 1500, 1750, 2000]
MAX_PROMPT_TYPES_PER_PES = 4
LINK_BANDWIDTHS = [10, 20, 30, 40]
LINK_WEIGHTS = (5, 20)

MIN_PROMPT_CHAIN_LENGTH = 3
MAX_PROMPT_CHAIN_LENGTH = 15
PROMPT_CHAIN_LENGTHS = (MIN_PROMPT_CHAIN_LENGTH, MAX_PROMPT_CHAIN_LENGTH)

PROMPT_COMP_DEMAND = (0.5, 0.8)
VPEP_COMP_DEMAND = 0.6
BANDWIDTH_DEMAND = [1, 2, 3, 4, 5]
COMP_LATENCY_PER_CORE = 0.01
PEP_VERIFY_LATENCY = 0.5
VPEP_VERIFY_LATENCY = 0.02
NUM_NETWORKS = 10
NUM_AI_SRS_PER_NETWORK = 1000

# --- Helper Functions ---
def generate_network():
    G = nx.erdos_renyi_graph(NUM_NODES, 0.3, directed=True)
    nodes = list(G.nodes)
    random.shuffle(nodes)
    num_peps = random.randint(*NUM_PEPS_RANGE)
    X = nodes[:NUM_AIGC_SERVERS]
    Q = nodes[NUM_AIGC_SERVERS:NUM_AIGC_SERVERS + num_peps]
    S = nodes[NUM_AIGC_SERVERS + num_peps:]
    for node in G.nodes:
        if node in S:
            G.nodes[node]['type'] = 'PES'
            G.nodes[node]['capacity'] = random.choice(PES_CAPACITIES)
            G.nodes[node]['prompt_types'] = random.sample(range(1, 8), MAX_PROMPT_TYPES_PER_PES)
        elif node in Q:
            G.nodes[node]['type'] = 'PEP'
        else:
            G.nodes[node]['type'] = 'AIGC'
    for u, v in G.edges():
        G.edges[u, v]['bandwidth'] = random.choice(LINK_BANDWIDTHS)
        G.edges[u, v]['weight'] = random.uniform(*LINK_WEIGHTS)
    return G, S, Q, X

def generate_ai_sr(G, S):
    chain_length = random.randint(*PROMPT_CHAIN_LENGTHS)
    PrC = list(range(chain_length))
    d_v = [2 for _ in PrC]
    BW = random.choice(BANDWIDTH_DEMAND)
    prompt_types = [random.randint(1, 7) for _ in PrC]
    c_v = [random.uniform(*PROMPT_COMP_DEMAND) for _ in PrC]
    source = random.choice(S)
    return {'PrC': PrC, 'd_v': d_v, 'BW': BW, 'prompt_types': prompt_types, 'c_v': c_v, 'source': source, 'length': chain_length}

def compute_transmission_latency(G, path, d_v, BW):
    if not path:
        return 0
    total_latency = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if (u, v) in G.edges():
            bw = G.edges[u, v]['bandwidth']
            latency = (d_v / (bw * 1000)) * 1000
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
    
    remaining_capacity = {n: G.nodes[n]['capacity'] for n in S}
    For_Path = [source]
    i = 1
    n_last = source
    PrC = PrC[1:]
    d_v = d_v[1:]
    prompt_types = prompt_types[1:]
    c_v = c_v[1:]
    deployment = {}
    
    while i < len(PrC) + 1:
        temp_k = []
        temp_T = float('inf')
        Sub_Path = []
        best_n = None
        
        for m in range(1, len(PrC) - i + 2):
            kappa = PrC[i-1:i-1+m]
            kappa_types = prompt_types[i-1:i-1+m]
            kappa_c_v = c_v[i-1:i-1+m]
            
            for n in S:
                can_host = all(pt in G.nodes[n]['prompt_types'] for pt in kappa_types)
                total_comp = sum(kappa_c_v)
                if n != n_last:
                    total_comp += VPEP_COMP_DEMAND
                if not can_host or total_comp > remaining_capacity[n]:
                    continue
                
                if n == n_last:
                    T_add = sum(c * COMP_LATENCY_PER_CORE for c in kappa_c_v)
                else:
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
                    
                    try:
                        path_direct = nx.shortest_path(G, n_last, n, weight='weight')
                        trans_direct = compute_transmission_latency(G, path_direct, d_v[i-2], BW)
                        vpep_latency = VPEP_VERIFY_LATENCY * d_v[i-2] + trans_direct
                    except nx.NetworkXNoPath:
                        vpep_latency = float('inf')
                    
                    min_verify_latency = min(pep_latencies + [vpep_latency]) if pep_latencies else vpep_latency
                    T_add = min_verify_latency + sum(c * COMP_LATENCY_PER_CORE for c in kappa_c_v)
                
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
                            for q in Q:
                                path_to_q = nx.shortest_path(G, n_last, q, weight='weight')
                                path_from_q = nx.shortest_path(G, q, n, weight='weight')
                                trans_to_q = compute_transmission_latency(G, path_to_q, d_v[i-2], BW)
                                trans_from_q = compute_transmission_latency(G, path_from_q, d_v[i-2], BW)
                                if trans_to_q + trans_from_q + PEP_VERIFY_LATENCY == min_verify_latency:
                                    Sub_Path = path_to_q[:-1] + path_from_q
                                    break
        
        if not temp_k:
            return None, float('inf'), {}, 0, 0, remaining_capacity
        
        for idx, p in enumerate(temp_k):
            deployment[p] = best_n
            remaining_capacity[best_n] -= c_v[i-1+idx]
        if best_n != n_last:
            remaining_capacity[n_last] -= VPEP_COMP_DEMAND
        
        i += len(temp_k)
        For_Path.extend(Sub_Path[1:] if Sub_Path[0] == n_last else Sub_Path)
        n_last = best_n
    
    min_latency = float('inf')
    best_x = None
    final_path = []
    for x in X:
        try:
            path_direct = nx.shortest_path(G, n_last, x, weight='weight')
            trans_direct = compute_transmission_latency(G, path_direct, d_v[-1], BW)
            vpep_latency = VPEP_VERIFY_LATENCY * d_v[-1] + trans_direct
            if vpep_latency < min_latency:
                min_latency = vpep_latency
                best_x = x
                final_path = path_direct
            
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
        return None, float('inf'), deployment, 0, 0, remaining_capacity
    
    For_Path.extend(final_path[1:] if final_path[0] == n_last else final_path)
    deployment[PrC[-1]] = best_x
    
    total_latency = 0
    verification_latency = 0
    num_verification_steps = 0
    for idx, p in enumerate(PrC):
        n = deployment[p]
        total_latency += c_v[idx] * COMP_LATENCY_PER_CORE
        if idx == 0:
            continue
        prev_n = deployment[PrC[idx-1]]
        if n != prev_n:
            path = []
            for j in range(len(For_Path)):
                if For_Path[j] == prev_n:
                    for k in range(j, len(For_Path)):
                        path.append(For_Path[k])
                        if For_Path[k] == n:
                            break
                    break
            trans_latency = compute_transmission_latency(G, path, d_v[idx-1], BW)
            uses_pep = any(node in Q for node in path[1:-1])
            if uses_pep:
                verify_latency = PEP_VERIFY_LATENCY
            else:
                verify_latency = VPEP_VERIFY_LATENCY * d_v[idx-1]
            total_latency += trans_latency + verify_latency
            verification_latency += verify_latency
            num_verification_steps += 1
    
    return For_Path, total_latency, deployment, verification_latency, num_verification_steps, remaining_capacity

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
    PrC = PrC[1:]
    d_v = d_v[1:]
    prompt_types = prompt_types[1:]
    c_v = c_v[1:]
    
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
            return None, float('inf'), deployment, 0, 0, remaining_capacity
        
        deployment[p] = best_n
        remaining_capacity[best_n] -= c_v[idx]
        path = nx.shortest_path(G, current, best_n, weight='weight')
        For_Path.extend(path[1:] if path[0] == current else path)
        current = best_n
    
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
        return None, float('inf'), deployment, 0, 0, remaining_capacity
    
    For_Path.extend(final_path[1:] if final_path[0] == current else final_path)
    deployment[PrC[-1]] = best_x
    
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
    
    return For_Path, total_latency, deployment, 0, 0, remaining_capacity

# --- PCDF Implementation (PCDO + Least-Latency Verification) ---
def pcdf(G, S, Q, X, ai_sr):
    For_Path, _, deployment, _, _, remaining_capacity = pcdo(G, S, X, ai_sr)
    if For_Path is None:
        return None, float('inf'), 0, 0
    
    PrC = ai_sr['PrC'][1:]
    d_v = ai_sr['d_v'][1:]
    BW = ai_sr['BW']
    c_v = ai_sr['c_v'][1:]
    
    new_path = [For_Path[0]]
    total_latency = 0
    verification_latency = 0
    num_verification_steps = 0
    for idx, p in enumerate(PrC):
        total_latency += c_v[idx] * COMP_LATENCY_PER_CORE
        if idx == 0:
            continue
        prev_n = deployment[PrC[idx-1]]
        n = deployment[p]
        if n != prev_n:
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
                verify_latency = PEP_VERIFY_LATENCY
            else:
                latency = vpep_latency
                verify_latency = VPEP_VERIFY_LATENCY * d_v[idx-1]
                path = path_direct
            
            total_latency += latency
            verification_latency += verify_latency
            num_verification_steps += 1
            new_path.extend(path[1:] if path[0] == new_path[-1] else path)
    
    current = new_path[-1]
    for x in X:
        if x in For_Path:
            path = nx.shortest_path(G, current, x, weight='weight')
            trans_direct = compute_transmission_latency(G, path, d_v[-1], BW)
            vpep_latency = VPEP_VERIFY_LATENCY * d_v[-1] + trans_direct
            total_latency += vpep_latency
            verification_latency += VPEP_VERIFY_LATENCY * d_v[-1]
            num_verification_steps += 1
            break
    
    return new_path, total_latency, verification_latency, num_verification_steps

# --- Main Experiment ---
def run_experiment():
    # Results dictionary: {method: {length: {metric: value}}}
    results = {
        'SCBB-L': {length: {'latency': [], 'accepted': 0, 'total': 0, 'runtime': [], 
                             'verification_latency': [], 'verification_steps': [], 
                             'path_length': [], 'resource_utilization': []} for length in range(MIN_PROMPT_CHAIN_LENGTH, MAX_PROMPT_CHAIN_LENGTH+1)},
        'PCDO': {length: {'latency': [], 'accepted': 0, 'total': 0, 'runtime': [], 
                          'verification_latency': [], 'verification_steps': [], 
                          'path_length': [], 'resource_utilization': []} for length in range(MIN_PROMPT_CHAIN_LENGTH, MAX_PROMPT_CHAIN_LENGTH+1)},
        'PCDF': {length: {'latency': [], 'accepted': 0, 'total': 0, 'runtime': [], 
                          'verification_latency': [], 'verification_steps': [], 
                          'path_length': [], 'resource_utilization': []} for length in range(MIN_PROMPT_CHAIN_LENGTH, MAX_PROMPT_CHAIN_LENGTH+1)}
    }
    
    for network_idx in range(NUM_NETWORKS):
        print(f"Processing Network {network_idx + 1}/{NUM_NETWORKS}...")
        G, S, Q, X = generate_network()
        
        for sr_idx in range(NUM_AI_SRS_PER_NETWORK):
            if (sr_idx + 1) % 500 == 0:
                print(f"  Network {network_idx + 1}: Processed {sr_idx + 1}/{NUM_AI_SRS_PER_NETWORK} AI-SRs")
            ai_sr = generate_ai_sr(G, S)
            chain_length = ai_sr['length']
            
            # Run SCBB-L
            start_time = time.time()
            path, latency, _, verif_latency, verif_steps, remaining_capacity = scbb_l(G, S, Q, X, ai_sr)
            runtime = time.time() - start_time
            results['SCBB-L'][chain_length]['total'] += 1
            results['SCBB-L'][chain_length]['runtime'].append(runtime)
            if path is not None:
                results['SCBB-L'][chain_length]['accepted'] += 1
                results['SCBB-L'][chain_length]['latency'].append(latency)
                results['SCBB-L'][chain_length]['verification_latency'].append(verif_latency)
                results['SCBB-L'][chain_length]['verification_steps'].append(verif_steps)
                results['SCBB-L'][chain_length]['path_length'].append(len(path) - 1)  # Number of hops
                # Compute resource utilization efficiency (coefficient of variation)
                used_capacity = [G.nodes[n]['capacity'] - remaining_capacity[n] for n in S]
                mean_capacity = np.mean(used_capacity) if used_capacity else 0
                std_capacity = np.std(used_capacity) if used_capacity else 0
                cv = std_capacity / mean_capacity if mean_capacity > 0 else float('inf')
                results['SCBB-L'][chain_length]['resource_utilization'].append(cv)
            
            # Run PCDO
            start_time = time.time()
            path, latency, _, verif_latency, verif_steps, remaining_capacity = pcdo(G, S, X, ai_sr)
            runtime = time.time() - start_time
            results['PCDO'][chain_length]['total'] += 1
            results['PCDO'][chain_length]['runtime'].append(runtime)
            if path is not None:
                results['PCDO'][chain_length]['accepted'] += 1
                results['PCDO'][chain_length]['latency'].append(latency)
                results['PCDO'][chain_length]['verification_latency'].append(verif_latency)
                results['PCDO'][chain_length]['verification_steps'].append(verif_steps)
                results['PCDO'][chain_length]['path_length'].append(len(path) - 1)
                used_capacity = [G.nodes[n]['capacity'] - remaining_capacity[n] for n in S]
                mean_capacity = np.mean(used_capacity) if used_capacity else 0
                std_capacity = np.std(used_capacity) if used_capacity else 0
                cv = std_capacity / mean_capacity if mean_capacity > 0 else float('inf')
                results['PCDO'][chain_length]['resource_utilization'].append(cv)
            
            # Run PCDF
            start_time = time.time()
            path, latency, verif_latency, verif_steps = pcdf(G, S, Q, X, ai_sr)
            runtime = time.time() - start_time
            results['PCDF'][chain_length]['total'] += 1
            results['PCDF'][chain_length]['runtime'].append(runtime)
            if path is not None:
                results['PCDF'][chain_length]['accepted'] += 1
                results['PCDF'][chain_length]['latency'].append(latency)
                results['PCDF'][chain_length]['verification_latency'].append(verif_latency)
                results['PCDF'][chain_length]['verification_steps'].append(verif_steps)
                results['PCDF'][chain_length]['path_length'].append(len(path) - 1)
                # Resource utilization: re-run pcdo to get remaining capacity
                _, _, _, _, _, remaining_capacity = pcdo(G, S, X, ai_sr)
                used_capacity = [G.nodes[n]['capacity'] - remaining_capacity[n] for n in S]
                mean_capacity = np.mean(used_capacity) if used_capacity else 0
                std_capacity = np.std(used_capacity) if used_capacity else 0
                cv = std_capacity / mean_capacity if mean_capacity > 0 else float('inf')
                results['PCDF'][chain_length]['resource_utilization'].append(cv)
    
    # Compute average metrics for each method and chain length
    metrics = {
        'latency': {}, 'acceptance_ratio': {}, 'runtime': {}, 
        'verification_overhead': {}, 'verification_steps': {},
        'path_length': {}, 'resource_utilization': {}
    }
    for method in results:
        metrics['latency'][method] = []
        metrics['acceptance_ratio'][method] = []
        metrics['runtime'][method] = []
        metrics['verification_overhead'][method] = []
        metrics['verification_steps'][method] = []
        metrics['path_length'][method] = []
        metrics['resource_utilization'][method] = []
        for length in range(MIN_PROMPT_CHAIN_LENGTH, MAX_PROMPT_CHAIN_LENGTH+1):
            data = results[method][length]
            # Average latency
            avg_latency = np.mean(data['latency']) if data['latency'] else float('inf')
            metrics['latency'][method].append(avg_latency)
            # Acceptance ratio
            acceptance_ratio = data['accepted'] / data['total'] if data['total'] > 0 else 0
            metrics['acceptance_ratio'][method].append(acceptance_ratio)
            # Average runtime (in ms)
            avg_runtime = np.mean(data['runtime']) * 1000 if data['runtime'] else float('inf')
            metrics['runtime'][method].append(avg_runtime)
            # Verification overhead ratio
            verif_overhead = []
            for lat, verif_lat in zip(data['latency'], data['verification_latency']):
                if lat > 0:
                    verif_overhead.append(verif_lat / lat)
            avg_verif_overhead = np.mean(verif_overhead) if verif_overhead else 0
            metrics['verification_overhead'][method].append(avg_verif_overhead)
            # Average number of verification steps
            avg_verif_steps = np.mean(data['verification_steps']) if data['verification_steps'] else 0
            metrics['verification_steps'][method].append(avg_verif_steps)
            # Average path length (hops)
            avg_path_length = np.mean(data['path_length']) if data['path_length'] else float('inf')
            metrics['path_length'][method].append(avg_path_length)
            # Average resource utilization efficiency (coefficient of variation)
            avg_resource_util = np.mean(data['resource_utilization']) if data['resource_utilization'] else float('inf')
            metrics['resource_utilization'][method].append(avg_resource_util)
    
    # Print final results
    print("\nFinal Results:")
    for method in results:
        print(f"{method}:")
        for length in range(MIN_PROMPT_CHAIN_LENGTH, MAX_PROMPT_CHAIN_LENGTH+1):
            print(f"  Chain Length {length}:")
            print(f"    Average Latency: {metrics['latency'][method][length-MIN_PROMPT_CHAIN_LENGTH]:.4f} ms")
            print(f"    Acceptance Ratio: {metrics['acceptance_ratio'][method][length-MIN_PROMPT_CHAIN_LENGTH]:.4f}")
            print(f"    Average Runtime: {metrics['runtime'][method][length-MIN_PROMPT_CHAIN_LENGTH]:.4f} ms")
            print(f"    Verification Overhead Ratio: {metrics['verification_overhead'][method][length-MIN_PROMPT_CHAIN_LENGTH]:.4f}")
            print(f"    Average Verification Steps: {metrics['verification_steps'][method][length-MIN_PROMPT_CHAIN_LENGTH]:.2f}")
            print(f"    Average Path Length (Hops): {metrics['path_length'][method][length-MIN_PROMPT_CHAIN_LENGTH]:.2f}")
            print(f"    Resource Utilization Efficiency (CV): {metrics['resource_utilization'][method][length-MIN_PROMPT_CHAIN_LENGTH]:.4f}")
    
    # Generate visualizations
    chain_lengths = list(range(MIN_PROMPT_CHAIN_LENGTH, MAX_PROMPT_CHAIN_LENGTH+1))
    methods = ['PCDO', 'SCBB-L', 'PCDF']
    bar_width = 0.25
    index = np.arange(len(chain_lengths))
    
    # Plot 1: Average Latency
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        plt.bar(index + i * bar_width, metrics['latency'][method], bar_width, label=method)
    plt.xlabel('Prompt Chain Length')
    plt.ylabel('Average Latency (ms)')
    plt.title('Average Latency by Prompt Chain Length')
    plt.xticks(index + bar_width, chain_lengths)
    plt.legend()
    plt.tight_layout()
    plt.savefig('latency_comparison.png')
    plt.close()
    
    # Plot 2: Acceptance Ratio
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        plt.bar(index + i * bar_width, metrics['acceptance_ratio'][method], bar_width, label=method)
    plt.xlabel('Prompt Chain Length')
    plt.ylabel('Acceptance Ratio')
    plt.title('Acceptance Ratio by Prompt Chain Length')
    plt.xticks(index + bar_width, chain_lengths)
    plt.legend()
    plt.tight_layout()
    plt.savefig('acceptance_ratio_comparison.png')
    plt.close()
    
    # Plot 3: Average Runtime
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        plt.bar(index + i * bar_width, metrics['runtime'][method], bar_width, label=method)
    plt.xlabel('Prompt Chain Length')
    plt.ylabel('Average Runtime (ms)')
    plt.title('Average Runtime by Prompt Chain Length')
    plt.xticks(index + bar_width, chain_lengths)
    plt.legend()
    plt.tight_layout()
    plt.savefig('runtime_comparison.png')
    plt.close()
    
    # Plot 4: Verification Overhead Ratio
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        plt.bar(index + i * bar_width, metrics['verification_overhead'][method], bar_width, label=method)
    plt.xlabel('Prompt Chain Length')
    plt.ylabel('Verification Overhead Ratio')
    plt.title('Verification Overhead Ratio by Prompt Chain Length')
    plt.xticks(index + bar_width, chain_lengths)
    plt.legend()
    plt.tight_layout()
    plt.savefig('verification_overhead_comparison.png')
    plt.close()
    
    # Plot 5: Average Number of Verification Steps
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        plt.bar(index + i * bar_width, metrics['verification_steps'][method], bar_width, label=method)
    plt.xlabel('Prompt Chain Length')
    plt.ylabel('Average Verification Steps')
    plt.title('Average Number of Verification Steps by Prompt Chain Length')
    plt.xticks(index + bar_width, chain_lengths)
    plt.legend()
    plt.tight_layout()
    plt.savefig('verification_steps_comparison.png')
    plt.close()
    
    # Plot 6: Average Path Length (Hops)
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        plt.bar(index + i * bar_width, metrics['path_length'][method], bar_width, label=method)
    plt.xlabel('Prompt Chain Length')
    plt.ylabel('Average Path Length (Hops)')
    plt.title('Average Path Length by Prompt Chain Length')
    plt.xticks(index + bar_width, chain_lengths)
    plt.legend()
    plt.tight_layout()
    plt.savefig('path_length_comparison.png')
    plt.close()
    
    # Plot 7: Resource Utilization Efficiency (Coefficient of Variation)
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        plt.bar(index + i * bar_width, metrics['resource_utilization'][method], bar_width, label=method)
    plt.xlabel('Prompt Chain Length')
    plt.ylabel('Resource Utilization Efficiency (CV)')
    plt.title('Resource Utilization Efficiency by Prompt Chain Length')
    plt.xticks(index + bar_width, chain_lengths)
    plt.legend()
    plt.tight_layout()
    plt.savefig('resource_utilization_comparison.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()