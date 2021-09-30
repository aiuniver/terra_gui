<template>
  <div class="t-texts">
    <div class="t-texts__header">
      <div v-for="(layer, i) of layers" :key="'layer_' + i" class="t-texts__block">
        <p>{{ layer }}</p>
        <t-field inline label="loss">
          <t-checkbox-new :value="true" name="loss" @change="change(layer, $event)" />
        </t-field>
        <t-field inline label="metrics">
          <t-checkbox-new :value="true" name="metrics" @change="change(layer, $event)" />
        </t-field>
      </div>
    </div>
    <div class="t-texts__content">
      <Table :data="data" :settings="settings" />
    </div>
    <div class="t-progress__overlay" v-if="!isData">
      <LoadSpiner v-if="isLearning" text="Загрузка данных..." />
    </div>
  </div>
</template>

<script>
import LoadSpiner from '@/components/forms/LoadSpiner';
import Table from './Table';
import { mapGetters } from 'vuex';
export default {
  name: 'Texts',
  components: {
    Table,
    LoadSpiner
  },
  data: () => ({
    settings: {},
    ops: {
      scrollPanel: {
        scrollingX: false,
        scrollingY: false,
      },
    },
  }),
  computed: {
    ...mapGetters({
      status: 'trainings/getStatus',
    }),
    isLearning() {
      return ['addtrain', 'training'].includes(this.status);
    },
    layers() {
      const obj = this.data?.['1']?.data || {};
      const layres = [];
      for (let key in obj) {
        layres.push(key);
      }
      return layres;
    },
    data() {
      return this.$store.getters['trainings/getTrainData']('progress_table') || {};
    },
    isData() {
      return Object.values(this.data).length;
    },
  },
  methods: {
    change(layer, { name, value }) {
      if (!this.settings[layer]) {
        this.settings[layer] = {};
      }
      this.settings[layer][name] = value;
      this.settings = { ...this.settings };
      console.log(this.settings);
    },
  },
  created() {
    this.layers.forEach(key => {
      this.settings[key] = {
        loss: true,
        metrics: true,
      };
    });
  },
};
</script>

<style lang="scss" scoped>
.t-texts {
  margin-bottom: 20px;
  position: relative;
  &__header {
    display: flex;
    gap: 15px;
    justify-self: start;
    align-items: center;
    padding: 5px 0;
    p {
      font-size: 14px;
    }
  }
  &__block {
    display: flex;
    gap: 10px;
    p {
      margin: 0 0 10px 0;
    }
  }
  &__content {
    display: flex;
    height: 100%;
  }
  &__empty {
    display: flex;
    height: 100%;
    justify-content: center;
    padding-top: 10px;
    font-size: 16px;
    opacity: 0.5;
    user-select: none;
  }
}
</style>

