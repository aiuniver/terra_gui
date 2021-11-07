<template>
  <div class="board">
    <scrollbar :ops="{ scrollPanel: { scrollingX: false } }">
      <div class="wrapper">
        <at-collapse :value="collapse" @on-change="change" class="mt-3">
          <at-collapse-item class="mt-3" title="Прогресс обучения" center>
            <Progress :outputs="outputs" :interactive="interactive" />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Метрики" center>
            <Graphs metric="metric_graphs" :outputs="outputs" :interactive="interactive" />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Лоссы" center>
            <Graphs metric="loss_graphs" :outputs="outputs" :interactive="interactive" />
          </at-collapse-item>

          <at-collapse-item class="mt-3" title="Промежуточные результаты" center>
            <!-- <PrePesults/> -->
            <!-- <Images /> -->
            <Prediction :outputs="outputs" :interactive="interactive" />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Таблица прогресса обучения" center>
            <Texts :outputs="outputs" :interactive="interactive" />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Статистические данные" center>
            <Stats :outputs="outputs" :interactive="interactive" />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Баланс данных" center style="position: relative;">
            <Balance :outputs="outputs" :interactive="interactive" />
          </at-collapse-item>
        </at-collapse>
      </div>
    </scrollbar>
    <LargeImage v-show="largeImg" />
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import Texts from './main/texts/index.vue';
import Progress from './main/progress/';
import Stats from './main/stats';
import Balance from './main/balance';
import Prediction from './main/prediction';
import Graphs from './main/chars/index';

export default {
  name: 'Graphics',
  components: {
    Prediction,
    Texts,
    // Images,
    Progress,
    Stats,
    Balance,
    Graphs,
    LargeImage: () => import('./main/prediction/components/LargeImage.vue'),
  },
  data: () => ({
    // collabse: [],
  }),
  computed: {
    ...mapGetters({
      // status: "trainings/getStatus",
      largeImg: 'trainings/getLargeImg'
    }),
    collapse: {
      set(value) {
        this.$store.dispatch('trainings/setСollapse', value);
      },
      get() {
        return this.$store.getters['trainings/getСollapse'];
      },
    },
    show() {
      return this.collabse.includes('0');
    },
    outputs() {
      return this.$store.getters['trainings/getOutputs'];
    },
    interactive() {
      return this.$store.getters['trainings/getInteractive'];
    },
  },
  methods: {
    change(e) {
      // this.$emit('collabse', this.collabse);
      this.collabse = e;
      console.log(e);
    },
  },
};
</script>

<style scoped>
.board {
  flex: 1 1 auto;
  overflow: hidden;
}
.wrapper {
  padding: 20px;
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
}
.mt-3 {
  width: 100%;
}
</style>
