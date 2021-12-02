<template>
  <main class="page-deploy">
    <div class="cont">
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
                      :default-layout="defaultLayout"
                      :type="type"
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
      <Params 
        :params="params"
        :height="height"
        :module-list="moduleList"
        :project-data="projectData"
        :user-data="userData"
        :sent-deploy="isSendParamsDeploy"
        :params-downloaded="paramsSettings"
        :overlay-status="overlay"
        @downloadSettings="downloadSettings"
        @overlay="setOverlay" 
        @sendParamsDeploy="sendParamsDeploy"
        @clear="clearParams"
      />
    </div>
    <div class="overlay" v-if="overlay"></div>
  </main>
</template>

<script>
export default {
  name: 'Datasets',
  components: {
    Params: () => import('@/components/deploy/params/Params'),
    IndexCard: () => import('./IndexCard'),
    Table: () => import('./Table.vue'),
    TableClass: () => import('./TableClass.vue'),
  },
  computed: {
    ...mapGetters({
      defaultLayout: 'deploy/getDefaultLayout',
      dataLoaded: 'deploy/getDataLoaded',
      cards: 'deploy/getCards',
      autoHeight: 'settings/autoHeight',
      type: 'deploy/getDeployType',
      deploy: 'deploy/getDeploy',
      params: 'deploy/getParams',
      height: 'settings/height',
      moduleList: 'deploy/getModuleList',
      projectData: 'projects/getProject',
      userData: 'projects/getUser',
    }),
    isTable() {
      return ['DataframeClassification', 'DataframeRegression'].includes(this.type);
    },
  },
  data: () => ({ 
    overlay: false,
    updateKey: 0,
    idCheckProgressSendDeploy: null,
    paramsSettings: {
      isSendParamsDeploy: false,
      isParamsSettingsLoad: false
    }
  }),
  methods: {
    clearParams(){
      this.$store.dispatch('deploy/clear');
    },
    setOverlay(value) {
      this.overlay = value;
    },
    async checkProgressSendDeploy() {
      let answer = await this.$store.dispatch('deploy/checkProgress');
      if (answer) {
        this.overlay = true
        this.paramsSettings.isSendParamsDeploy = true
        clearTimeout(this.idCheckProgressSendDeploy)
      }
    },
    async sendParamsDeploy(){
      const res = await this.$store.dispatch('deploy/sendDeploy', data);
      if (res) {
        const { error, success } = res;
        if (!error && success) {
          this.overlay = true
          this.idCheckProgressSendDeploy = setTimeout(this.checkProgressSendDeploy, 2000);
        }
      }
    },
    async downloadSettings({ type = null, name = null }){
      if (type && name) {
        this.overlay = true
        const res = await this.$store.dispatch('deploy/downloadSettings', { type, name });
        if (res?.success) {
          this.overlay = false
          this.paramsSettings.isParamsSettingsLoad = true
        }
      }
    },
    async reload(index) {
      this.updateKey++;
      await this.$store.dispatch('deploy/reloadCard', [String(index)]);
    },
    async reloadAll() {
      let indexes = [];
      for (let i = 0; i < this.cards.length; i++) indexes.push(String(i));
      await this.$store.dispatch('deploy/reloadCard', indexes);
    },
  }
};
</script>

<style lang="scss" scoped>
.page-deploy {
  height: 100%;
  width: 100%;
}
.cont {
  background: #17212b;
  padding: 0;
  display: flex;
  height: 100%;
}
.overlay {
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background-color: #0e1621;
  opacity: 0.45;
}

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