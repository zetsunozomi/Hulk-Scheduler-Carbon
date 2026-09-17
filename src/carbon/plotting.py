"""Static scientific figures; rendering never changes selections or summaries."""

import os
from pathlib import Path
import tempfile
import textwrap

from .common import ContractError


COLORS = {'Fixed-4':'#7F7F7F','Fixed-8':'#4A6B8A','Fixed-16':'#6B5B95','Fixed-32':'#B07C35',
          'Fixed-64':'#4A6B8A','Fixed-128':'#B07C35',
          'Fixed-Mix':'#009E73','Plan-once':'#56B4E9','Rollout-MPC':'#D55E00','ScaleDown':'#0072B2',
          'Queue-blind-MPC':'#CC79A7','Current-CI':'#8C564B','Predictor-advised':'#8C564B','Precommitted-RL':'#8C564B',
          'ScaleDown-lower-endpoint':'#CC79A7','ScaleDown-upper-endpoint':'#009E73'}
MAIN_METHODS = ('Fixed-4','Fixed-8','Fixed-16','Fixed-32','Fixed-64','Fixed-128','Fixed-Mix','Plan-once','Rollout-MPC','ScaleDown')


def label(method):
    return method.replace('ScaleDown-lower-endpoint','Lower-endpoint policy').replace('ScaleDown-upper-endpoint','Upper-endpoint policy')


def _header(fig,data,beta,title):
    fig.text(.07,.93,title,fontsize=13,fontweight='bold',color='#15283B')
    panel = '\n'.join(textwrap.wrap(data['panel'],85))
    fig.text(.07,.885,f"{panel}  |  budget = {beta:g} x T_ref",fontsize=9,color='#445363',va='top')
    if data['purpose']!='research':
        fig.text(.93,.985,data['purpose'].upper()+' - SOFTWARE / DEVELOPMENT ONLY',ha='right',
                 va='top',fontsize=7.4,fontweight='bold',color='#A13D2D')


def _axes_style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(color='#E6EAF0',linewidth=.6,zorder=0)
    ax.tick_params(labelsize=8)
    ax.set_axisbelow(True)


def _main_page(plt,data,beta):
    fig = plt.figure(figsize=(7.4,5.0))
    _header(fig,data,beta,'E2  Cost and deadline outcomes')
    left = fig.add_axes([.10,.26,.43,.51]);right = fig.add_axes([.74,.26,.21,.51])
    _axes_style(left);_axes_style(right)
    groups = [next((g for g in data['groups'] if g['beta']==beta and g['method']==m),None) for m in MAIN_METHODS]
    groups = [g for g in groups if g is not None]
    any_intervals = False
    missing = []
    for index,group in enumerate(groups):
        method = group['method'];color = COLORS[method]
        marker = '*' if method=='ScaleDown' else 'D' if method=='Fixed-Mix' else 's' if method.startswith('Fixed-') else 'o'
        supported = group['status']=='supported'
        stats = group['statistics']
        if stats is not None:
            point = stats['point'];x,y = point['mean_tat_hours'],point['normalized_carbon']
            intervals = stats['intervals']
            if intervals:
                # Draw intervals as endpoints directly: percentile intervals need
                # not contain the point estimate, so negative error magnitudes
                # must not be clipped into a misleading centered error bar.
                lx,hx = intervals['mean_tat_hours'];ly,hy = intervals['normalized_carbon']
                left.plot([lx,hx],[y,y],color=color,linewidth=1,alpha=.7)
                left.plot([x,x],[ly,hy],color=color,linewidth=1,alpha=.7)
                any_intervals = True
            if len(group['seed_points'])>1:
                left.scatter([p['mean_tat_hours'] for p in group['seed_points']],
                             [p['normalized_carbon'] for p in group['seed_points']],s=12,color=color,alpha=.38,zorder=3)
            left.scatter([x],[y],s=110 if method=='ScaleDown' else 40,marker=marker,
                         facecolors=color if supported else 'white',edgecolors=color,linewidths=1.2,zorder=4)
        else:
            missing.append(method)
        summary = group['summary']
        if summary:
            lo,hi = summary['miss_lower'],summary['miss_upper']
            right.plot([lo,hi],[index,index],color=color,linewidth=3,alpha=.5,zorder=2)
            right.scatter([(lo+hi)/2],[index],s=24,color=color,zorder=3)
            deadlines = [r['deadline'] for r in group['seed_records']]
            if all(bound and bound['support_enabled'] for bound in deadlines):
                bound = max(r['confidence_upper'] for r in deadlines)
                right.scatter([bound],[index],s=25,marker='D',facecolors='white',edgecolors=color,zorder=3)
        else:
            right.text(.02,index,'unavailable',fontsize=7,color='#6B7280',va='center')
    left.set_xlabel('Mean turnaround time (hours)',fontsize=9)
    left.set_ylabel(f"Modeled carbon / training Fixed-4\n(rho = {data['nominal_rho']:g})",fontsize=9)
    left.set_xlim(left=max(0,left.get_xlim()[0]));left.set_ylim(bottom=max(0,left.get_ylim()[0]))
    right.set_yticks(range(len(groups)),[label(g['method']) for g in groups],fontsize=7.3)
    right.set_ylim(len(groups)-.5,-.5)
    right.set_xlim(-.04,1.06);right.set_xticks([0,.5,1.])
    right.axvline(data['epsilon'],color='#333333',linestyle=':',linewidth=1)
    right.set_xlabel('Deadline-miss probability',fontsize=8)
    right.set_title(f"Tolerance = {data['epsilon']:g}",fontsize=9,pad=9)
    for tick,group in zip(right.get_yticklabels(),groups):tick.set_color(COLORS[group['method']])
    fig.text(.10,.16,'Filled: validation/test miss bounds pass. Open: no bound support. Small dots: individual seeds.',fontsize=7.1,color='#445363')
    fig.text(.10,.125,'Miss panel: dot/range = observed miss bounds; diamond = largest available seed upper bound.',fontsize=7.1,color='#445363')
    level = 100*(1-data['statistics']['settings']['alpha'])
    note = (f'Bars: {level:g}% marginal paired calendar intervals; RL also resamples seeds.' if any_intervals else
            'Intervals unavailable: insufficient calendar blocks or no development dependence audit.')
    fig.text(.10,.09,note,fontsize=7.1,color='#445363')
    if missing:fig.text(.10,.055,'Full carbon unavailable: '+', '.join(missing),fontsize=7.1,color='#A13D2D')
    return fig


def _power_page(plt,data,beta):
    fig = plt.figure(figsize=(7.4,4.9))
    _header(fig,data,beta,'E4  Sensitivity to the assumed power model')
    ax = fig.add_axes([.11,.29,.80,.48]);_axes_style(ax)
    lo,hi = data['rho_interval']
    groups = [g for g in data['groups'] if g['beta']==beta and (g['method']=='ScaleDown' or
              (g['method'].startswith('ScaleDown-') and g['method'].endswith('-endpoint')))]
    curves = [g for g in groups if g.get('power_curve')]
    styles = {'ScaleDown':'-','ScaleDown-lower-endpoint':'--','ScaleDown-upper-endpoint':':'}
    endpoint_intervals = False
    for group in curves:
        method = group['method'];color = COLORS.get(method,'#8C564B')
        main = [p for p in group['power_curve'] if p['rho']>=lo]
        ideal = next((p for p in group['power_curve'] if p['rho']==0),None)
        ax.plot([p['rho'] for p in main],[p['ratio_of_means'] for p in main],label=label(method)+(' ['+group['status']+']' if group['status']!='supported' else ''),
                color=color,linestyle=styles.get(method,'--'),linewidth=1.8)
        if ideal and lo>0:ax.scatter([0],[ideal['ratio_of_means']],color=color,marker='x',s=32,zorder=4)
        intervals = group['statistics']['intervals']
        if intervals:
            for index,rho in enumerate((lo,hi)):
                bounds = intervals.get(f'endpoint_ratio_{index}')
                if bounds:
                    ax.plot([rho,rho],bounds,color=color,linewidth=1.3,zorder=4)
                    ax.plot([rho,rho],bounds,linestyle='none',marker='_',color=color,markersize=6,zorder=4)
                    endpoint_intervals = True
        root = group['power']['break_even_rho']
        if root is not None and lo<=root<=hi:
            ax.scatter([root],[1.],facecolors='white',edgecolors=color,s=40,zorder=4)
    ax.axhline(1,color='#4B5563',linestyle=':',linewidth=1)
    if lo>0:
        ax.axvspan(0,lo,facecolor='#F3F4F6',zorder=0)
        ax.text(lo/2,.03,'rho=0 stress point\n(no curve in this gap)',transform=ax.get_xaxis_transform(),
                ha='center',fontsize=7,color='#6B7280')
    ax.set_xlim(-.025,hi+.025)
    ax.set_xticks(sorted(set([0,lo,(lo+hi)/2,hi])))
    ax.set_xlabel('Assumed power parameter rho',fontsize=9)
    ax.set_ylabel('Mean carbon / frozen comparator',fontsize=9)
    if curves:ax.legend(loc='best',frameon=False,fontsize=8)
    else:ax.text(.5,.55,'Full-cohort power comparison unavailable',transform=ax.transAxes,ha='center',fontsize=10,color='#6B7280')
    comparator = next((g['comparator_method'] for g in groups if g['comparator_method']),None)
    fig.text(.11,.18,'Comparator: '+str(comparator or 'unavailable')+'. Policies and comparator stay fixed across rho.',fontsize=7.4,color='#445363')
    fig.text(.11,.14,'Curves vary a power assumption; they are not measured power or statistical confidence bands.',fontsize=7.4,color='#445363')
    level = 100*(1-data['statistics']['settings']['alpha'])
    note = f'Endpoint bars: {level:g}% marginal paired intervals with seed resampling; joint evidence is in the report.' if endpoint_intervals else 'Endpoint intervals unavailable; curves are descriptive point estimates only.'
    fig.text(.11,.10,note,fontsize=7.4,color='#445363')
    unavailable = [label(g['method']) for g in groups if not g.get('power_curve')]
    if unavailable:
        fig.text(.11,.06,'No complete curve: '+', '.join(unavailable),fontsize=7.4,color='#A13D2D')
    elif not any(g['method']!='ScaleDown' for g in groups):
        fig.text(.11,.06,'Endpoint-trained counterpart is not in the frozen plan at this budget.',fontsize=7.4,color='#A13D2D')
    return fig


def render_figures(data,output):
    # Respect a configured cache. Otherwise keep font-cache writes in task temp,
    # not an unwritable home directory or the tracked repository.
    with tempfile.TemporaryDirectory(prefix='carbon-mpl-') as cache:
        previous = {key:os.environ.get(key) for key in ('MPLCONFIGDIR','XDG_CACHE_HOME')}
        for key,value in previous.items():
            if value is None:os.environ[key] = cache
        try:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_pdf import PdfPages
            except ImportError:
                raise ContractError('Figure export requires matplotlib; install requirements-plots.txt in CARBON_PYTHON or use --tables-only') from None
            with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
                                 'axes.linewidth':.7,'savefig.facecolor':'white','text.usetex':False}):
                for name,make in [('E2-main',_main_page),('E4-power',_power_page)]:
                    with PdfPages(Path(output)/(name+'.pdf'),metadata={'Title':name+' - '+data['panel'],
                                   'Subject':data['purpose']+'; derived from sealed experiment logs','Creator':'carbon result export'}) as pdf:
                        for index,beta in enumerate(data['budgets'],1):
                            fig = make(plt,data,beta)
                            pdf.savefig(fig)
                            fig.savefig(Path(output)/f'{name}-page-{index:02d}.png',dpi=180)
                            plt.close(fig)
        finally:
            for key,value in previous.items():
                if value is None:os.environ.pop(key,None)
