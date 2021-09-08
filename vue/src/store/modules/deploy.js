import temp from "../temp/deploy";
import {defLayout, originaltextStyle} from "../const/deploy"
export default {
    namespaced: true,
    state: () => ({
      graphicData: temp.data,
      defaultLayout: defLayout,
      origTextStyle: originaltextStyle,
      Cards: [
        {
          type: "card",
          original: {
            type: "image",
            imgUrl: '1.jpg'
          },
          result: {
            type: "image",
            imgUrl: 'segmentation/1.jpg'
          }
        },
        {
          type: "card",
           original: {
            type: "image",
            imgUrl: '2.jpg'
          },
          result: {
            type: "image",
            imgUrl: 'segmentation/2.jpg'
          }
        },
        {
          type: "card",
       original: {
            type: "image",
            imgUrl: '3.jpg'
          },
          result: {
            type: "image",
            imgUrl: 'segmentation/3.jpg'
          }
        },
                {
          type: "card",
        original: {
            type: "image",
            imgUrl: '4.jpg'
          },
          result: {
            type: "image",
            imgUrl: 'segmentation/4.jpg'
          }
        },
        {
          type: "card",
          original: {
            type: "image",
            imgUrl: '5.jpg'
          },
          result: {
            type: "image",
            imgUrl: 'segmentation/5.jpg'
          }
        },
                {
          type: "card",
          original: {
            type: "image",
            imgUrl: '6.jpg'
          },
          result: {
            type: "image",
            imgUrl: 'segmentation/6.jpg'
          }
        },
                {
          type: "card",
          original: {
            type: "image",
            imgUrl: '7.jpg'
          },
          result: {
            type: "image",
            imgUrl: 'segmentation/7.jpg'
          }
        },
                {
          type: "card",
          original: {
            type: "image",
            imgUrl: '8.jpg'
          },
          result: {
            type: "image",
            imgUrl: 'segmentation/8.jpg'
          }
        },
                {
          type: "card",
          original: {
            type: "image",
            imgUrl: '9.jpg'
          },
          result: {
            type: "image",
            imgUrl: 'segmentation/9.jpg'
          }
        },
                {
          type: "card",
          original: {
            type: "image",
            imgUrl: '10.jpg'
          },
          result: {
            type: "image",
            imgUrl: 'segmentation/10.jpg'
          }
        },
      ],
      moduleList: {
        api_text: "",
        url: "",
      }
    }),
    mutations: {
      SET_MODULE_LIST(state, value) {
        state.moduleList = { ...state.moduleList, ...value};
      },
    },
    actions: {
      async SendDeploy({ dispatch }, data) {
        return await dispatch('axios', { url: '/deploy/upload/', data: data }, { root: true });
      },
      async CheckProgress({ commit, dispatch }) {
        const { data } = await dispatch('axios', { url: '/deploy/upload/progress/'}, { root: true });
        // console.log(data)
        if(data.finished){
          commit("SET_MODULE_LIST", data.data);
        }
        return data.finished;
      },
    },
    getters: {
      getModuleList: ({ moduleList }) => moduleList,
      getGraphicData: ({ graphicData }) => graphicData,
      getDefaultLayout: ({ defaultLayout }) => defaultLayout,
      getOrigTextStyle: ({ origTextStyle }) => origTextStyle,
      getCards: ({ Cards }) => Cards,
    }
}