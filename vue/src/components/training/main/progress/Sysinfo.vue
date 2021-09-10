<template>
  <div class="t-sysinfo">
      <div class="t-sysinfo__label">
          Информация об устройстве
      </div>
      <div class="t-sysinfo__grid">
          <div class="t-sysinfo__grid-item">GPU</div>
          <div class="t-sysinfo__grid-item">
              <p class="pu-name">NVIDIA GeForce GTX 1060 6 GB</p>
              <p>1.00% (1.28GB / 6.00GB)</p>
              <div class="t-sysinfo__progress-bar">
                  <div class="t-sysinfo__progress-bar--fill" style="width: 100px;"></div>
              </div>
          </div>
          <div class="t-sysinfo__grid-item">RAM</div>
          <div class="t-sysinfo__grid-item">
              <p>44.00% (14.07GB / 31.95GB)</p>
              <div class="t-sysinfo__progress-bar">
                  <div class="t-sysinfo__progress-bar--fill" style="width: 220px;"></div>
              </div>
          </div>
          <div class="t-sysinfo__grid-item">Disk</div>
          <div class="t-sysinfo__grid-item warning">
              <p>96.20% (96.21GB / 100.00GB)</p>
              <div class="t-sysinfo__progress-bar">
                  <div class="t-sysinfo__progress-bar--fill" style="width: 266px;"></div>
              </div>
          </div>
      </div>
  </div>
</template>

<script>
export default {

}
</script>

<style lang="scss" scoped>
.t-sysinfo {
    display: flex;
    flex-direction: column;
    color: #A7BED3;
    gap: 30px;
    font-size: 14px;
    .pu-name {
        color: inherit;
    }
    p {
        color: #fff;
    }
    &__grid {
        display: grid;
        row-gap: 17px;
        column-gap: 30px;
        grid-template: repeat(3, max-content) / min-content 290px;
        &-item {
            &.warning p {
                color: #CA5035;
            }
            &.warning div div {
                background: #CA5035;
            }
        }
    }
    &__progress-bar {
        height: 5px;
        width: 100%;
        background: #242F3D;
        border-radius: 2.5px;
        margin-top: 5px;
        &--fill {
            background: #65CA35;
            border-radius: 2.5px;
            height: 100%;
        }
    }
}
</style>