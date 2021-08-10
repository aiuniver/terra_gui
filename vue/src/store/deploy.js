export default {
    namespaced: true,
    state: () => ({
        moduleList: [
            {
                title: "Распознавание птиц на видео",
                url: "https://demo.neural-university.ru/birds_video/",
                input: "od_content файл с изображением",
                output: "Относительная ссылка"
            }
        ]
    }),
    mutations: {},
    actions: {},
    getters: {
        getModuleList: ({ moduleList }) => moduleList,
    }
}