export type Dataset = 'iris' | 'wine' | 'ecoli';
export type Status = 'idle' | 'pending' | 'fulfilled' | 'rejected';


export interface DataInstance {
    uid: number
    features: number[]
    target: string
    projection?: number[]
}

export interface ProjectedDataInstance extends DataInstance {
    projection: number[]
}

export type Data = DataInstance[]
export type ProjectedData = ProjectedDataInstance[]
