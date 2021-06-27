export default {
  namespaced: true,
  state: () => ({
    data: {
      centerX: 1024,
      centerY: 140,
      scale: 1,
      nodes: [
        {
          id: 1,
          x: -650,
          y: -100,
          type: "input",
          label: "test1",
        },
        {
          id: 2,
          x: -500,
          y: 50,
          type: "action",
          label: "test2",
        },
        {
          id: 3,
          x: -800,
          y: 50,
          type: "action",
          label: "test3",
        },
        {
          id: 4,
          x: -650,
          y: 150,
          type: "output",
          label: "test4",
        },
      ],
      links: [
        {
          id: 1,
          from: 1,
          to: 2,
        },
        {
          id: 2,
          from: 1,
          to: 3,
        },
        {
          id: 3,
          from: 2,
          to: 4,
        },
        {
          id: 4,
          from: 3,
          to: 4,
        },
      ],
    },
    
  }),
  mutations: {
    SET_DATA(state, value) {
      state.data = [...value];
    }
  },
  actions: {
    get({ commit }) {
      console.log(commit);
      // try {
      //   const { data } = await axios.get(
      //     "https://60d20d1f5b017400178f5047.mockapi.io/api/v1/datasets"
      //   );
      //   commit("SET_DATASETS", data);
      // } catch (error) {
      //   console.log(error);
      // }
    },
  },
  getters: {
    getData({ data }) {
      return data;
    },
  }
};
