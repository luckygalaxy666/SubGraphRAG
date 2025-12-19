import argparse
import csv
from pathlib import Path

import igraph as ig
import leidenalg


def load_edges_tsmc(path: Path):
    edges = []
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            head, _, rel, tail = row[0], row[1], row[2], row[3]
            time_val = row[5] if len(row) > 5 else ''
            edges.append({
                'head': head,
                'tail': tail,
                'relation': rel,
                'time': time_val,
                'score': ''
            })
    return edges


def load_edges_result(path: Path):
    edges = []
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            head = row.get('head')
            tail = row.get('tail')
            rel = row.get('relation')
            if not head or not tail or rel is None:
                continue
            edges.append({
                'head': head,
                'tail': tail,
                'relation': rel,
                'time': row.get('time', ''),
                'score': row.get('score', '')
            })
    return edges


def load_edges_auto(path: Path):
    """Auto-detect format based on header/file name."""
    with path.open(newline='', encoding='utf-8') as f:
        peek = next(csv.reader(f), [])
    header_lower = [c.lower() for c in peek]
    if 'head' in header_lower and 'relation' in header_lower:
        return load_edges_result(path)
    if '_sub' in path.stem:
        return load_edges_result(path)
    return load_edges_tsmc(path)


def build_graph(edges):
    verts = {}
    v_list = []
    for edge in edges:
        h, t = edge['head'], edge['tail']
        if h not in verts:
            verts[h] = len(v_list)
            v_list.append(h)
        if t not in verts:
            verts[t] = len(v_list)
            v_list.append(t)
    g = ig.Graph()
    g.add_vertices(v_list)
    g.add_edges([(verts[e['head']], verts[e['tail']]) for e in edges])
    g.es['relation'] = [e['relation'] for e in edges]
    return g


def run_leiden(g, resolution=1.0):
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights=None
    )
    return part


def visualize(g, partition, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
    colors = [palette[partition.membership[v.index]] for v in g.vs]
    layout = g.layout('fr')
    ig.plot(
        g,
        target=str(out_file),
        layout=layout,
        vertex_size=12,
        vertex_label=None,  # 不展示节点名称，避免图过于拥挤
        vertex_color=colors,
        edge_label=None,
        bbox=(1200, 800),
        margin=40
    )
    print(f'Saved plot to {out_file}')


def main():
    parser = argparse.ArgumentParser(
        description='Run Leiden community detection on tsmc CSV triples and visualize.')
    parser.add_argument('--tsmc_csv', type=str, nargs='*', default=[],
                        help='Raw tsmc triple CSVs (head,type,rel,tail,type,time).')
    parser.add_argument('--result_csv', type=str, nargs='*', default=[],
                        help='Sub CSVs with head,relation,tail,time,score.')
    parser.add_argument('--auto_csv', type=str, nargs='*', default=[],
                        help='CSV paths with auto format detection (raw or sub).')
    parser.add_argument('--out_dir', type=str, default='outputs/tsmc_leiden',
                        help='Directory to save visualization.')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Leiden resolution parameter.')
    parser.add_argument('--export_neo4j_csv', action='store_true',
                        help='Also export node and edge CSVs for Neo4j import.')
    args = parser.parse_args()

    # 如果未指定输入，则默认读取 data_files/tsmc 下的所有 CSV（便于直接处理台积电获得英伟达AI芯片订单.csv 等原始文件）
    if not args.tsmc_csv and not args.result_csv:
        default_dir = Path('data_files/tsmc')
        args.tsmc_csv = [str(p) for p in default_dir.glob('*.csv')]

    edges = []
    for p in args.tsmc_csv:
        edges.extend(load_edges_tsmc(Path(p)))
    for p in args.result_csv:
        edges.extend(load_edges_result(Path(p)))
    for p in args.auto_csv:
        edges.extend(load_edges_auto(Path(p)))

    if not edges:
        raise ValueError('No edges loaded. Please provide at least one CSV via --tsmc_csv or --result_csv.')

    g = build_graph(edges)
    partition = run_leiden(g, args.resolution)

    stem = 'leiden'
    if args.result_csv:
        stem = Path(args.result_csv[0]).stem + '_leiden'
    elif args.tsmc_csv:
        stem = Path(args.tsmc_csv[0]).stem + '_leiden'
    out_dir = Path(args.out_dir)
    out_file = out_dir / f'{stem}.png'
    visualize(g, partition, out_file)

    if args.export_neo4j_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        node_csv = out_dir / f'{stem}_nodes.csv'
        edge_csv = out_dir / f'{stem}_edges.csv'
        with node_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'community'])
            for v, c in zip(g.vs['name'], partition.membership):
                writer.writerow([v, c])
        with edge_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['head', 'tail', 'relation', 'time', 'score'])
            for e in edges:
                writer.writerow([e['head'], e['tail'], e['relation'], e.get('time', ''), e.get('score', '')])
        print(f'Saved Neo4j import CSVs:\n  nodes: {node_csv}\n  edges: {edge_csv}')
        print('Neo4j import example:\n'
              "LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row\n"
              "MERGE (n:Entity {name: row.name})\n"
              "SET n.community = toInteger(row.community);\n"
              "LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row\n"
              "MERGE (h:Entity {name: row.head})\n"
              "MERGE (t:Entity {name: row.tail})\n"
              "MERGE (h)-[r:REL {rel: row.relation}]->(t)\n"
              "SET r.time = row.time, r.score = toFloat(row.score);")


if __name__ == '__main__':
    main()
