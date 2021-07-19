<template>
  <div class="board">
    <div class="wrapper">
      <Filters />
      <div class="project-datasets-block datasets">
        <div class="title" @click="click('name')">Выберите датасет</div>
        <vue-custom-scrollbar class="scroll-area" :settings="settings" :style="height">
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
        </vue-custom-scrollbar>
      </div>
    </div>
  </div>
</template>

<script>
import vueCustomScrollbar from "vue-custom-scrollbar";
import "vue-custom-scrollbar/dist/vueScrollbar.css";
import Filters from "@/components/datasets/Filters.vue";
import { mapGetters } from "vuex";
import Card from "@/components/datasets/Card";

export default {
  components: {
    Card,
    Filters,
    vueCustomScrollbar,
  },
  data: () => ({
    settings: {
      suppressScrollY: false,
      suppressScrollX: true,
      wheelPropagation: false,
    },
  }),
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

<style >
.scroll-area {
  position: relative;
  width: 100%;
  /* height: 400px; */
}
.board {
  border-right: black solid 1px;
}
</style>