import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('paper')
sns.set_palette('muted')

### start by plotting the two combined

full_rms = pd.read_csv('/home/clionaodoherty/associations/60min_rms_for_plotting.csv', index_col=0)
full_r2 = pd.read_csv('/home/clionaodoherty/associations/fulltime_r2_for_plotting.csv',index_col=0)

# convert rms secs to mins for plotting
full_rms['lagmins'] = full_rms['lagsecs'] / 60

sns.lineplot(data=full_rms, x='lagmins', y='-rms_normed')
sns.lineplot(data=full_r2, x='lagmins', y='r2_normed')

plt.xlabel('lag (min)')
plt.ylabel('mean normalised metric')
plt.xlim((-2.5,50))
plt.legend(['perceptual dropoff \n (-RMS similarity)','object associations \n (R2 score of autocorrelation)'])
sns.despine()
plt.tight_layout()

plt.savefig('dropoff_combined.png')
plt.close()

## then plot the first minute of the RMS to show pattern better
firstmin_rms = pd.read_csv('/home/clionaodoherty/associations/first60sec_rms_for_plotting.csv',index_col=0)
sns.lineplot(data=firstmin_rms, x='lag', y='-rms_normed')
sns.lineplot(data=full_r2[full_r2['lagsecs']<=180], x='lagsecs',y='r2_normed')

plt.xlabel('lag (sec)')
plt.ylabel('mean normalised metric')
plt.xlim((-2.5,60))
plt.legend(['perceptual dropoff \n (-RMS similarity)','object associations \n (R2 score of autocorrelation)'])
sns.despine()
plt.tight_layout()

plt.savefig('dropoff_combined_firstmin.png')
plt.close()

## plot both on separate axes beside each other
fig, (ax1,ax2) = plt.subplots(ncols=2)

sns.lineplot(data=firstmin_rms, x='lag', y='-rms_normed', ax=ax1)
ax1.set_title('perceptual similarity')
ax1.set_xlabel('lag (sec)')
ax1.set_ylabel('-RMS similarity \n (mean normalised)')
ax1.set_ylim((-1.0,3.0))
ax1.set_xlim((-2.5,50))
ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, size=20, weight='bold')

sns.lineplot(data=full_r2, x='lagmins', y='r2_normed', ax=ax2)
ax2.set_title('object associations')
ax2.set_xlabel('lag (min)')
ax2.set_ylabel('R2 score of autocorrelation \n (mean normalised)')
ax2.set_ylim((-1.0,3.0))
ax2.set_xlim((-2.5,50))
ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, size=20, weight='bold')

sns.despine()
plt.tight_layout()

plt.savefig('dropoff_sidebyside.png')
plt.close()


