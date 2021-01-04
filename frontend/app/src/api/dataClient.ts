import {API_URL} from "./common";

async function dataClient<T>(path: string, config: RequestInit): Promise<T> {
    const request = new Request(`${API_URL}${path}`, config);
    const response = await fetch(request);

    if (!response.ok) {
        throw new Error(response.statusText);
    }
    return response.json().catch(() => ({}));
}

export async function getData<T>(path: string, config?: RequestInit): Promise<T> {
    const init = {method: 'get', ...config};

    return await dataClient<T>(path, init);
}
