import type { StatusResponse } from "@/types/api";
import { PUBLIC_BACKEND_URL } from "@/consts/const";

export async function get_experiment_status(id: number){
    const res = await fetch(`${PUBLIC_BACKEND_URL}/experiment/${id}/status`);
    return await res.json() as StatusResponse;
}