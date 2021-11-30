<template>
  <div class="board">
    <scrollbar>
      <div class="wrapper">
        <div class="content">
          <div class="board__data-field">
            <div>
              <div class="board__title">Исходные данные / Предсказанные данные</div>
              <div v-if="!isTable" class="board__data">
                <IndexCard
                  v-for="(card, i) in cards"
                  :key="'#board-card-' + i"
                  :card="card"
                  :color-map="deploy.color_map"
                  :index="i"
                  @reload="reload"
                />
              </div>
              <div v-else class="board__data">
                <Table
                  v-if="type === 'DataframeRegression'"
                  v-bind="deploy"
                  :key="'#board-' + updateKey"
                  @reload="reload"
                  @reloadAll="reloadAll"
                />
                <TableClass
                  v-if="type === 'DataframeClassification'"
                  v-bind="deploy"
                  :key="'#board-' + updateKey"
                  @reload="reload"
                  @reloadAll="reloadAll"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';

export default {
  components: {
    IndexCard: () => import('./IndexCard'),
    Table: () => import('./Table.vue'),
    TableClass: () => import('./TableClass.vue'),
  },
  data: () => ({ updateKey: 0 }),
  computed: {
    ...mapGetters({
      dataLoaded: 'deploy/getDataLoaded',
      cards: 'deploy/getCards',
      height: 'settings/autoHeight',
      type: 'deploy/getDeployType',
      deploy: 'deploy/getDeploy',
    }),
    isTable() {
      return ['DataframeClassification', 'DataframeRegression'].includes(this.type);
    },
  },
  methods: {
    async reload(index) {
      this.updateKey++
      await this.$store.dispatch('deploy/reloadCard', [String(index)]);
    },
    async reloadAll() {
      let indexes = [];
      for (let i = 0; i < this.cards.length; i++) indexes.push(String(i));
      await this.$store.dispatch('deploy/reloadCard', indexes);
    },
  },
};
</script>

<style lang="scss" scoped>
.board {
  flex: 1 1 auto;
  overflow: hidden;
  &__data {
    display: flex;
    flex-wrap: wrap;
  }
  &__title {
    font-size: 12px;
    line-height: 24px;
    color: #a7bed3;
  }
  &__table {
    padding: 20px 0 0 0;
  }
  &__load-data {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    button {
      width: 156px;
      font-size: 14px;
      line-height: 24px;
    }
  }
  &__reload-all {
    display: flex;
    width: 174px;
    padding: 8px 10px 10px 10px;
    margin-bottom: 30px;
    justify-content: center;
    align-items: center;
    i {
      width: 16px;
    }
    span {
      font-size: 14px;
      line-height: 24px;
      padding-left: 8px;
    }
  }
}
.wrapper {
  padding: 20px;
  height: 100%;
}
</style>