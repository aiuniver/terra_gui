<template>
  <div class="board">
    <div class="wrapper">
      <h1>Cтраница в разработке</h1>
      <h3></h3>

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
}
</style>