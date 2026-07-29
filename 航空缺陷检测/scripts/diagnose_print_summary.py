import json, os
p = r'd:\元强化学习\代码\航空缺陷检测\reports\diagnose\cross_domain_distill.json'
d = json.load(open(p, encoding='utf-8'))
print('Domains:', list(d['domains'].keys()))
for dn, dv in d['domains'].items():
    agg = dv.get('aggregate', {})
    cls = list(dv.get('per_class', {}).keys())
    print(f'{dn}: n={dv["total_samples"]} classes={cls}')
    print(f'  aggregate: {agg}')
    for c, items in dv.get('per_class', {}).items():
        Vs = [x['color']['V_mean'] for x in items]
        Gs = [x['color']['gray_mean'] for x in items]
        Es = [x['target_size'].get('edge_density', 0) for x in items]
        if Vs:
            import statistics as st
            print(f'  {c}: n={len(items)} V={st.mean(Vs):.1f}±{st.pstdev(Vs):.1f} gray={st.mean(Gs):.1f}±{st.pstdev(Gs):.1f} edge={st.mean(Es):.3f}')
print('Cross diff:', d.get('cross_domain_diff'))
