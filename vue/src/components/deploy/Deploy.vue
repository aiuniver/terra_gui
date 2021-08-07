<template>
  <div class="board">
    <div class="wrapper">
      <div class="load-data">
        <button>Загрузить данные</button>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";

export default {
  computed: {
    ...mapGetters({
      datasets: "datasets/getDatasets",
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
  text-align: center;
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
</style>