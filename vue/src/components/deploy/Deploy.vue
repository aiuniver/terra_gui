<template>
  <div class="board">
    <div class="wrapper">
      <div class="content" v-if="dataLoaded">
        <button class="reload-all">Перезагрузить все</button>
        <div class="data-field">
          <div class="title">Исходные данные / Предсказанные данные</div>
          <div class="data">
            <IndexCard v-for="card in Cards" :key="card" :DataBlock="card"/>
          </div>
        </div>
      </div>

      <div class="load-data" v-if="!dataLoaded">
        <button @click="$store.dispatch('deploy/setDataLoaded', !dataLoaded)">Загрузить данные</button>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
import IndexCard from "./IndexCard";

export default {
  components: {
    IndexCard
  },
  data: () => ({
    loaded: false,
    Cards: [
      {
        original: {
          type: "image",
          data: " "
        },
        result: {
          type: "image",
          data: ""
        }
      },
      {
        original: {
          type: "image",
          data: " "
        },
        result: {
          type: "image",
          data: ""
        }
      },
      {
        original: {
          type: "image",
          data: " "
        },
        result: {
          type: "image",
          data: ""
        }
      },
      {
        original: {
          type: "image",
          data: " "
        },
        result: {
          type: "image",
          data: ""
        }
      },
      {
        original: {
          type: "image",
          data: " "
        },
        result: {
          type: "image",
          data: ""
        }
      },
      {
        original: {
          type: "image",
          data: " "
        },
        result: {
          type: "text",
          data: "Дерево"
        }
      },
      {
        original: {
          type: "image",
          data: ""
        },
        result: {
          type: "text",
          data: "Дерево"
        }
      },
      {
        original: {
          type: "image",
          data: ""
        },
        result: {
          type: "text",
          data: "Дерево"
        }
      },
      {
        original: {
          type: "image",
          data: ""
        },
        result: {
          type: "text",
          data: "Дерево"
        }
      },
      {
        original: {
          type: "image",
          data: ""
        },
        result: {
          type: "text",
          data: "Дерево"
        }
      },
    ]
  }),
  computed: {
    ...mapGetters({
      dataLoaded: "deploy/getDataLoaded",
      height: "settings/autoHeight",
    }),
  },
  methods: {
    click(dataset){
      console.log(dataset)
    }
  },
};
</script>

<style lang="scss" scoped>
.board {
  flex-shrink: 1;
  width: 100%;
}
.wrapper{
  padding: 50px;
  height: 100%;
}
.load-data{
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  button{
    width: 156px;
    font-size: 14px;
    line-height: 24px;
  }
}
.reload-all{
  width: 174px;
}
.data-field{
  padding-top: 30px;
  .title{
    font-size: 12px;
    line-height: 24px;
    color: #A7BED3;
  }
}
.data{
  display: flex;
  flex-wrap: wrap;
}

</style>