import pandas

ser2 =pd.Series([1,0,2,2,0])
ser3 = pd.Series([1,1,0,0,3])
ser4 = pd.Series([0,0,1,1,0])
ser5 = pd.Series([2,3,0,1,0])
ser6 = pd.Series([0,3,1,0,1])
df = pd.concat([ser2, ser3,ser4,ser5,ser6], axis=1)

def make_symm(df, idx1, idx2):
    a=idx1
    b=idx2
    if df.iloc[a][b] != df.iloc[b][a] and df.iloc[a][b]!=0:
        df.iloc[b][a] = df.iloc[a][b]
    if df.iloc[b][a] != df.iloc[a][b] and df.iloc[b][a]!=0:
        df.iloc[a][b] = df.iloc[b][a]
    return df

for i in range(0,len(df)+1):
    for j in range(0,len(df)+1):
        make_symm(lev_df,i,j)


    '''
    print('begin symmetry')
    for k1, v1 in leverage_dict.items():
        for k2, v2 in leverage_dict.items():
            if k1[0] == k2[1] and k1[1] == k2[0] and not v1 == 0:
                leverage_dict[k2] = v1
    print('complete symmetry')
    '''
    