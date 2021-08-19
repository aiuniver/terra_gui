<template>
  <div class="board">
    <div class="wrapper">
      <div class="content">
<!--        <button class="board__reload-all">-->
<!--          <i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i>-->
<!--          <span>Перезагрузить все</span>-->
<!--        </button>-->
        <div class="board__data-field">
          <div class="board__title">Исходные данные / Предсказанные данные</div>
          <div class="board__data">
            <IndexCard v-for="(card, i) in Cards" :key="'card-'+i" v-bind="card"/>
          </div>
        </div>
<!--        <Table/>-->
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
import IndexCard from "./IndexCard";
// import Table from "./Table";
export default {
  components: {
    IndexCard
    // Table
  },
  data: () => ({
    loaded: false,
    Cards: [
      {
        type: "graphic"
      },
      {
        type: "graphic"
      },
      {
        type: "graphic"
      },
      {
        type: "graphic"
      },
    ],
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
.board__load-data{
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
.board__reload-all{
  display: flex;
  width: 174px;
  padding: 8px 10px 10px 10px;
  justify-content: center;
  align-items: center;
  i{
    width: 16px;
  }
  span{
    font-size: 14px;
    line-height: 24px;
    padding-left: 8px;
  }
}
.board__data-field{
  padding-top: 30px;
}
.board__title{
    font-size: 12px;
    line-height: 24px;
    color: #A7BED3;
  }
.board__data{
  display: flex;
  flex-wrap: wrap;
}

</style>