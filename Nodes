graph = {
    1: {2: (10, 15, 1.2), 6: (10, 15, 1.2)},
    2: {1: (10, 15, 1.2), 3: (12, 25, 1.5), 5: (12, 25, 1.5), 6: (10, 15, 1.2)},
    3: {2: (12, 25, 1.5), 4: (12, 25, 1.5), 5: (14, 25, 1.5)},
    4: {}, # Sink node
    5: {2: (12, 25, 1.5), 3: (12, 25, 1.5), 4: (14, 25, 1.2), 6: (10, 25, 1.5)},
    6: {1: (10, 15, 1.2), 2: (10, 15, 1.2), 3: (10, 25, 1.3), 4: (10, 25, 1.5), 5: (10, 25, 1.5)}
}

def analyze_node(node_id):
    if node_id not in graph:
        print("Invalid Node ID.")
        return

    neighbors = graph[node_id]
    if not neighbors:
        print(f"Node {node_id} is a terminal/sink node (no outgoing connections).")
        return

    print(f"\n--- Analysis for Node {node_id} ---")
    print(f"{'Target':<8} | {'Dist (D)':<10} | {'Time (T)':<10} | {'Fuel (F)':<10}")
    print("-" * 45)

    d_vals, t_vals, f_vals = [], [], []

    for neighbor, (d, t, f) in neighbors.items():
        print(f"{neighbor:<8} | {d:<10} | {t:<10} | {f:<10}")
        d_vals.append(d)
        t_vals.append(t)
        f_vals.append(f)
    print("-" * 45)
    print(f"Stats      | Min/Max D  | Min/Max T  | Min/Max F")
    print(f"Low        | {min(d_vals):<10} | {min(t_vals):<10} | {min(f_vals):<10}")
    print(f"High       | {max(d_vals):<10} | {max(t_vals):<10} | {max(f_vals):<10}")

def main():
    while True:
        choice = input("\nEnter Node ID (1-6) or 'q' to quit: ")
        if choice.lower() == 'q':
            break
        try:
            analyze_node(int(choice))
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()