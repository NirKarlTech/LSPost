from pathlib import Path
path = Path('LS_Post_UI.py')
text = path.read_text(encoding='utf-8')
count = 0
new = []
for line in text.splitlines(True):
    if 'st.plotly_chart(' in line and 'key=' not in line:
        prefix = line[:line.index('st.plotly_chart')]
        if 'fig_matsum' in line:
            new.append(prefix + 'st.plotly_chart(fig_matsum, use_container_width=True, key=f"plotly_matsum_{count}")\n')
        elif 'fig_calc' in line:
            new.append(prefix + 'st.plotly_chart(fig_calc, use_container_width=True, key=f"plotly_calc_{count}")\n')
        elif 'fig_gc' in line:
            new.append(prefix + 'st.plotly_chart(fig_gc, use_container_width=True, key=f"plotly_gc_{count}")\n')
        else:
            new.append(prefix + 'st.plotly_chart(fig, use_container_width=True, key=f"plotly_{count}")\n')
        count += 1
    else:
        new.append(line)
path.write_text(''.join(new), encoding='utf-8')
print('replaced', count, 'plotly_chart calls')
