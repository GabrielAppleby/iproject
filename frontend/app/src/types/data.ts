export type Dataset = 'iris';
export type Projection = 'umap'
export type Status = 'idle' | 'pending' | 'fulfilled' | 'rejected';


export interface DataInstance {
    uid: number
    features: number[]
    target: string
    projections?: number[]
}

export interface ProjectedDataInstance extends DataInstance {
    projections: number[]
}

export type Data = DataInstance[]
export type ProjectedData = ProjectedDataInstance[]
