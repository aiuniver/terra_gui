<template>
  <div class="board">
    <div class="wrapper">
      <Filters />
      <div class="project-datasets-block datasets" style="padding-top: 173px">
        <div class="title">Выберите датасет</div>

        <vue-custom-scrollbar
          class="scroll-area"
          :settings="settings"
        >
          <div class="inner">
            <div class="dataset-card-container">
              <div class="dataset-card-wrapper">
                <template v-for="({ name, size, tags, date }, key) of datasets">
                  <Card
                    :name="name"
                    :size="size"
                    :tags="tags"
                    :date="date"
                    :key="key"
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
    }),
  },
};
</script>

<style >
.scroll-area {
  position: relative;
  width: 100%;
  height: 400px;
}
</style>