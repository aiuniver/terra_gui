<template>
  <div class="t-scatters">
    <div class="t-scatters__header">
      <div class="checks">
        <template v-for="(item, key, i) of statisticData">
          <t-checkbox
            v-if="item.labels"
            :key="'check_' + i"
            :inline="true"
            :label="`Выход ${key}`"
            :value="true"
            :name="key"
            @change="change($event, key)"
          />
        </template>
        <t-checkbox :inline="true" label="Автоотбновление" />
      </div>
      <button @click="showContent = !showContent">Показать</button>
    </div>
    <div v-if="showContent" class="t-scatters__content">
      <template v-for="(output, key, i) of statisticData">
        <Matrix v-if="output.data_array && isShowKeys.includes(key)" v-bind="output" :key="i" />
      </template>
    </div>
  </div>
</template>

<script>
import Matrix from './Matrix.vue';

export default {
  name: 't-scatters',
  components: {
    Matrix,
  },
  computed: {
    statisticData() {
      return this.$store.getters['trainings/getTrainData']('statistic_data') || [];
    },
  },
  data: () => ({
    showContent: true,
    isShowKeys: [],
  }),
  methods: {
    change(e, key) {
      console.log(e)
      this.isShowKeys = !this.isShowKeys.includes(key)
        ? [...this.isShowKeys, key]
        : this.isShowKeys.filter(item => item !== key);
    },
  },
  created() {
    this.isShowKeys = Object.keys(this.statisticData)
  }
};
</script>

<style lang="scss" scoped>
.t-scatters {
  &__header {
    display: flex;
    .checks {
      display: flex;
      flex-wrap: wrap;
      flex-shrink: 0;
      max-width: 320px;
      * {
        flex: 0 0 150px;
      }
    }
    button {
      flex: 0 0 150px;
    }
  }
  &__content {
    display: flex;
    flex-wrap: wrap;
    gap: 50px;
  }
}
</style>