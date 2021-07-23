<template>
  <div class="board">
    <div class="wrapper">
      <Filters />
      <div class="project-datasets-block datasets">
        <div class="title" @click="click('name')">Выберите датасет</div>
        <scrollbar :style="height">
          <div class="inner">
            <div class="dataset-card-container">
              <div class="dataset-card-wrapper">
                <template v-for="(dataset, key) of datasets">
                  <Card
                    :dataset="dataset"
                    :key="key"
                    @clickCard="click"
                  />
                </template>
              </div>
            </div>
          </div>
        </scrollbar>
      </div>
    </div>
  </div>
</template>

<script>
import Filters from "@/components/datasets/Filters.vue";
import { mapGetters } from "vuex";
import Card from "@/components/datasets/Card";

export default {
  components: {
    Card,
    Filters,
  },
  computed: {
    ...mapGetters({
      datasets: "datasets/getDatasets",
      height: "settings/autoHeight",
    }),
  },
  methods: {
    click(value){
      this.$store.dispatch('messages/setMessage', { message: `Выбран датасет «${value}»`})
    }
  }
};
</script>

<style lang="scss" scoped>
.board {
  flex-shrink: 1;
  width: 100%;
}
</style>